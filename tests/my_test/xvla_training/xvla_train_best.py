#!/usr/bin/env python
# ============================================================================
# X-VLA 训练入口：每次 eval_loss 刷新最低时，就地独立保存一份 best 模型
# ============================================================================
# 用法（与原入口一致，直接复用同一份 yaml，无需改动配置）：
#   python xvla_train_best.py --config_path ./xvla_demo_config_best.yaml
#   python xvla_train_best.py --config_path ./xvla_demo_config_best.yaml --steps=500
#
# 行为（两条保存线互相独立）：
#   1) 周期保存：完全沿用框架原逻辑，按 save_freq 保存到 checkpoints/step_N/，
#      并刷新 checkpoints/last 软链接 —— 不受本文件影响。
#   2) best 保存：训练循环每次 eval 都会在 main 进程打印一行
#      "step N: eval_loss=X.XXXX"（见 lerobot/scripts/lerobot_train.py 的 eval 分支）。
#      本入口解析这行日志，一旦发现 eval_loss 刷新了历史最低，就在那一步
#      **立刻真实落盘**一份 best 到 checkpoints/best/（覆盖式，只保留最优那一份）。
#      —— 因此不再要求 eval_steps == save_freq：最优步即使不是保存步也会被独立保存。
#      —— 之所以用日志解析：eval_loss 是 train() 的局部变量，外部拿不到，而本入口
#         又要求不改源码；该日志行是 main 进程必然输出的，最稳。
#   3) 训练结束后写 checkpoints/best_model.json，记录 best_step / best_eval_loss。
#
# 原理（全部 monkey-patch，改的都是 lerobot_train 模块里的名字，不动 lerobot 源码）：
#   · 复用 xvla_train_eval_split.py 已有的 patch（domain_id 透传、独立 eval_dataset、
#     放宽 eval_split 校验），因此配置文件无需任何改动。
#   · 包装 make_train_eval_datasets / make_pre_post_processors / update_policy，
#     分别捕获 cfg、preprocessor/postprocessor、policy/accelerator，供 best 保存使用。
#   · 包装 init_logging（它会 clear root handlers），在其后挂上日志解析 handler。
#
# 产出：
#   <output_dir>/checkpoints/best/           真实目录（覆盖式），含 pretrained_model/
#       └── pretrained_model/                权重 + config + pre/postprocessor
#   <output_dir>/checkpoints/best_model.json
#   用最优模型做评测/部署：
#     --policy.path=<output_dir>/checkpoints/best/pretrained_model
#
# 注意：
#   · best 只保存「可推理/可评测」所需内容（不含 optimizer / training_state），
#     体积远小于周期 checkpoint；如需从 best 继续训练请改用周期 checkpoint。
#   · 多卡分片（FSDP2）场景下，save_pretrained 是集合操作，无法只由主进程独立保存，
#     此时会打印 warning 并跳过 best 落盘（周期 checkpoint 不受影响）。
# ============================================================================

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from lerobot.distributed import is_main_process
from lerobot.distributed.checkpoint import is_sharded_module
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR

# 先复用原入口的全部 patch（domain_id / 独立 eval_dataset / 放宽校验）
import xvla_train_eval_split as _base  # noqa: F401  (导入即生效其 monkey-patch)
import lerobot.scripts.lerobot_train as _lt

logger = logging.getLogger("xvla_train_best")

_BEST_DIR_NAME = "best"
_BEST_INFO_NAME = "best_model.json"

# --------------------------------------------------------------------------- #
# 1) best 跟踪器（记录最优指标 + 捕获到的训练对象）
# --------------------------------------------------------------------------- #
_EVAL_LOSS_RE = re.compile(r"step\s+(\d+)\s*:\s*eval_loss=([-+0-9.eE]+)")


class _BestTracker:
    """记录 eval_loss 最小的 step，并持有 best 保存所需的训练对象引用。"""

    def __init__(self) -> None:
        self.best_step: int | None = None
        self.best_eval_loss: float | None = None
        # 捕获到的训练对象（由下面的 monkey-patch 填充）
        self.cfg = None
        self.policy = None
        self.accelerator = None
        self.preprocessor = None
        self.postprocessor = None
        self.warned_unsupported = False

    def observe(self, step: int, eval_loss: float) -> bool:
        """记录一次 eval。返回本次是否刷新了最优。"""
        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_step = step
            self.best_eval_loss = eval_loss
            return True
        return False


_tracker = _BestTracker()


# --------------------------------------------------------------------------- #
# 2) 保存 best 快照（覆盖式真实目录）
# --------------------------------------------------------------------------- #
def _can_save_main_only(policy, accelerator) -> bool:
    """判断能否只由主进程调用 save_pretrained（分片场景下它是集合操作，会挂死）。"""
    if accelerator is None:
        return True
    if accelerator.num_processes == 1:
        return True
    try:
        return not is_sharded_module(accelerator.unwrap_model(policy))
    except Exception:  # noqa: BLE001 - 判定失败时保守跳过
        return False


def _reset_dir(path: Path) -> None:
    """把 path 变成一个干净的空目录（兼容历史遗留的软链接 / 旧目录）。"""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_best_info(output_dir: Path) -> None:
    info = {
        "step": _tracker.best_step,
        "eval_loss": _tracker.best_eval_loss,
        "checkpoint": f"{CHECKPOINTS_DIR}/{_BEST_DIR_NAME}/{PRETRAINED_MODEL_DIR}",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    info_path = output_dir / CHECKPOINTS_DIR / _BEST_INFO_NAME
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_best_snapshot() -> None:
    """把当前 policy 独立保存为 best（覆盖上一份）。仅在 main 进程调用。"""
    t = _tracker
    if t.cfg is None or t.policy is None or t.preprocessor is None or t.postprocessor is None:
        logger.warning("[xvla_train_best] 训练对象尚未就绪，跳过本次 best 保存")
        return
    if not _can_save_main_only(t.policy, t.accelerator):
        if not t.warned_unsupported:
            logger.warning(
                "[xvla_train_best] 检测到分片(FSDP)多卡训练：best 无法仅由主进程独立保存，"
                "已跳过 best 落盘（周期 checkpoint 与 last 不受影响）。"
            )
            t.warned_unsupported = True
        return

    output_dir = Path(t.cfg.output_dir)
    best_dir = output_dir / CHECKPOINTS_DIR / _BEST_DIR_NAME
    pretrained_dir = best_dir / PRETRAINED_MODEL_DIR
    _reset_dir(best_dir)

    unwrapped_policy = t.accelerator.unwrap_model(t.policy) if t.accelerator is not None else t.policy
    unwrapped_policy.save_pretrained(pretrained_dir)
    t.cfg.save_pretrained(pretrained_dir)
    t.preprocessor.save_pretrained(pretrained_dir)
    t.postprocessor.save_pretrained(pretrained_dir)
    _write_best_info(output_dir)

    logger.info(
        f"[xvla_train_best] 已保存 best -> {pretrained_dir} "
        f"(step {t.best_step}, eval_loss={t.best_eval_loss:.4f})"
    )


# --------------------------------------------------------------------------- #
# 3) 日志解析：抓取 "step N: eval_loss=X.XXXX"，刷新最优时即时保存
# --------------------------------------------------------------------------- #
class _EvalLossHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if not is_main_process():
            return
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - 日志格式化失败不应影响训练
            return
        m = _EVAL_LOSS_RE.search(msg)
        if m is None:
            return
        try:
            step = int(m.group(1))
            eval_loss = float(m.group(2))
        except ValueError:
            return
        if not _tracker.observe(step, eval_loss):
            return
        logger.info(f"[xvla_train_best] new best eval_loss={eval_loss:.4f} at step {step}")
        try:
            _save_best_snapshot()
        except Exception:  # noqa: BLE001 - best 保存失败不能影响训练
            logger.exception("[xvla_train_best] 保存 best 失败（训练继续，周期 checkpoint 不受影响）")


_eval_loss_handler = _EvalLossHandler(level=logging.INFO)


# init_logging() 会先清空 root logger 的 handlers，必须在它之后再把 handler 挂上。
_original_init_logging = _lt.init_logging


def _patched_init_logging(*args, **kwargs):
    _original_init_logging(*args, **kwargs)
    root = logging.getLogger()
    if _eval_loss_handler not in root.handlers:
        root.addHandler(_eval_loss_handler)


_lt.init_logging = _patched_init_logging


# --------------------------------------------------------------------------- #
# 4) 捕获训练对象：cfg / preprocessor / postprocessor / policy / accelerator
# --------------------------------------------------------------------------- #
# cfg：make_train_eval_datasets(cfg) 在主/非主进程都会以完整 cfg 调用
_original_make_train_eval_datasets = _lt.make_train_eval_datasets


def _patched_make_train_eval_datasets(cfg):
    _tracker.cfg = cfg
    return _original_make_train_eval_datasets(cfg)


_lt.make_train_eval_datasets = _patched_make_train_eval_datasets

# preprocessor / postprocessor：make_pre_post_processors 返回 (pre, post)
_original_make_pre_post_processors = _lt.make_pre_post_processors


def _patched_make_pre_post_processors(*args, **kwargs):
    preprocessor, postprocessor = _original_make_pre_post_processors(*args, **kwargs)
    _tracker.preprocessor = preprocessor
    _tracker.postprocessor = postprocessor
    return preprocessor, postprocessor


_lt.make_pre_post_processors = _patched_make_pre_post_processors

# policy / accelerator：update_policy 每步都会以这两个对象调用
_original_update_policy = _lt.update_policy


def _patched_update_policy(
    train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator=None,
    lr_scheduler=None, lock=None, sample_weighter=None,
):
    _tracker.policy = policy
    _tracker.accelerator = accelerator
    return _original_update_policy(
        train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
        lr_scheduler=lr_scheduler, lock=lock, sample_weighter=sample_weighter,
    )


_lt.update_policy = _patched_update_policy


# --------------------------------------------------------------------------- #
# 5) 训练结束：log 汇总（best_model.json 已在每次刷新最优时写好）
# --------------------------------------------------------------------------- #
_original_train = _lt.train


def _finalize_best() -> None:
    if not is_main_process():
        return
    if _tracker.best_step is None:
        logger.info("[xvla_train_best] 本次训练未产生 eval（eval_steps=0 或没有 eval 数据），无 best 模型")
        return
    if _tracker.warned_unsupported:
        return
    output_dir = _tracker.cfg.output_dir if _tracker.cfg is not None else "?"
    logger.info(
        f"[xvla_train_best] best model: step {_tracker.best_step}, "
        f"eval_loss={_tracker.best_eval_loss:.4f} -> "
        f"{output_dir}/{CHECKPOINTS_DIR}/{_BEST_DIR_NAME}/{PRETRAINED_MODEL_DIR}"
    )


def _patched_train(*args, **kwargs):
    try:
        return _original_train(*args, **kwargs)
    finally:
        _finalize_best()


_lt.train = _patched_train


if __name__ == "__main__":
    # 完全复用原入口的 main()：解析 --config_path + 透传其余 CLI 参数
    _base.main()
