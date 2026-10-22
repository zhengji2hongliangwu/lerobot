#!/usr/bin/env python
# ============================================================================
# X-VLA 自定义训练入口（支持 yaml 顶层 domain_id 透传）
# ============================================================================
# 用法：
#   python xvla_train.py --config_path ./xvla_demo_config.yaml
#   python xvla_train.py --config_path ./xvla_demo_config.yaml --steps=500
#
# 工作原理：
#   1. 用 yaml.safe_load 读 yaml 顶层的 `domain_id` 字段（X-VLA 跨本体核心）
#   2. 从 yaml 临时删掉该字段，避免 draccus 严格 schema 报未知字段
#   3. monkey-patch lerobot_train 模块里的 make_pre_post_processors 引用，
#      在调用时把 domain_id 注入 preprocessor_overrides["xvla_add_domain_id"]
#      ——该 step 的 registry_name 见 policy_preprocessor.json
#   4. monkey-patch lerobot_train 模块里的 make_train_eval_datasets 引用，
#      当 yaml 顶层提供了 `eval_dataset` 块时，用独立数据集做离线评估
#      （否则回落到默认的同源 episode 切分逻辑）
#   5. monkey-patch TrainPipelineConfig.validate，当提供了顶层 `eval_dataset` 时，
#      跳过 “eval_steps>0 必须 dataset.eval_split>0” 的校验（独立 eval 集不需要切分）
#   6. 重写 sys.argv 让 lerobot_train.train() 走标准流程，透传其余 CLI 参数
#
# 不动 lerobot 源码即可让 soft_prompt_hub 查表取到正确的 domain 行，并支持独立 eval 数据集。
#
# 独立 eval 数据集用法（yaml 顶层）：
#   eval_dataset:
#     repo_id: local/eval_data
#     root: /mnt/data2/jizheng/datasets/eval_data
#   只需保证 eval_steps > 0；dataset.eval_split 保持 0.0 即可（独立 eval 集不做同源切分）。
# ============================================================================

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import IMAGENET_STATS

# 必须在调用 train() 之前 import + patch
import lerobot.scripts.lerobot_train as _lt

logger = logging.getLogger("xvla_train")

# 全局 domain_id，由 monkey-patch 闭包捕获
_DOMAIN_ID: int | None = None

# 保留原始 make_pre_post_processors 引用
_original_make_pre_post_processors = _lt.make_pre_post_processors


def _patched_make_pre_post_processors(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    """Inject domain_id into preprocessor_overrides before delegating to the original factory."""
    if _DOMAIN_ID is not None and pretrained_path is not None:
        overrides = kwargs.get("preprocessor_overrides")
        if overrides is None:
            overrides = {}
            kwargs["preprocessor_overrides"] = overrides
        # registry_name 来自 xvla-base/policy_preprocessor.json 的 step
        overrides["xvla_add_domain_id"] = {"domain_id": int(_DOMAIN_ID)}
        logger.info(
            f"[xvla_train] injected domain_id={_DOMAIN_ID} into preprocessor_overrides "
            f"for pretrained_path={pretrained_path}"
        )
    return _original_make_pre_post_processors(
        policy_cfg,
        pretrained_path=pretrained_path,
        pretrained_revision=pretrained_revision,
        **kwargs,
    )


# 应用 monkey-patch：替换 lerobot_train 模块本地的 make_pre_post_processors 名字
_lt.make_pre_post_processors = _patched_make_pre_post_processors


# 顶层 `eval_dataset` 块，由下面的 make_train_eval_datasets patch 使用
_EVAL_DATASET_CFG: dict | None = None

# 保留原始 make_train_eval_datasets 引用（同源 episode 切分逻辑）
_original_make_train_eval_datasets = _lt.make_train_eval_datasets


def _patched_make_train_eval_datasets(cfg):
    """当 yaml 提供了顶层 `eval_dataset` 块时，用独立数据集做 eval；否则回落到默认同源切分。"""
    if _EVAL_DATASET_CFG is None:
        return _original_make_train_eval_datasets(cfg)

    if not isinstance(_EVAL_DATASET_CFG, dict) or not _EVAL_DATASET_CFG.get("repo_id"):
        raise ValueError(
            "yaml 顶层 `eval_dataset` 必须是包含 `repo_id` 的映射，例如：\n"
            "eval_dataset:\n  repo_id: local/eval_data\n  root: /path/to/eval_data"
        )

    eval_repo_id = _EVAL_DATASET_CFG["repo_id"]
    eval_root = _EVAL_DATASET_CFG.get("root")
    eval_repo_type = _EVAL_DATASET_CFG.get("repo_type", cfg.dataset.repo_type)
    eval_revision = _EVAL_DATASET_CFG.get("revision", cfg.dataset.revision)

    # 训练集用全量数据集（不再按 eval_split 切分，避免训练数据被 eval 抽走）
    train_dataset = make_dataset(cfg)

    eval_meta = LeRobotDatasetMetadata(
        eval_repo_id,
        root=eval_root,
        revision=eval_revision,
        repo_type=eval_repo_type,
    )
    eval_dataset = LeRobotDataset(
        eval_repo_id,
        root=eval_root,
        delta_timestamps=resolve_delta_timestamps(cfg.trainable_config, eval_meta, cfg.rename_map),
        image_transforms=None,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=eval_revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=eval_repo_type,
    )

    if cfg.dataset.use_imagenet_stats:
        for key in eval_dataset.meta.camera_keys:
            if key in eval_dataset.meta.depth_keys:
                continue
            eval_dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                eval_dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    logger.info(
        f"[xvla_train] using separate eval dataset: repo_id={eval_repo_id}, "
        f"root={eval_root}, episodes={eval_dataset.num_episodes}"
    )
    return train_dataset, eval_dataset


# 应用 monkey-patch：替换 lerobot_train 模块本地的 make_train_eval_datasets 名字
_lt.make_train_eval_datasets = _patched_make_train_eval_datasets


# 保留原始 TrainPipelineConfig.validate 引用
_original_validate = TrainPipelineConfig.validate


def _patched_validate(self) -> None:
    """提供独立 `eval_dataset` 时，不再要求 dataset.eval_split > 0。

    TrainPipelineConfig 原本要求 eval_steps>0 就必须 eval_split>0（同源切分留出验证集）；
    使用独立 eval 数据集时该值不参与切分，也无意义。这里在调用原校验前临时抬高
    eval_split 以通过校验，随后还原，保证配置里仍保留用户填写的 0.0。
    """
    try:
        if (
            _EVAL_DATASET_CFG is not None
            and self.eval_steps > 0
            and self.dataset.eval_split == 0.0
        ):
            self.dataset.eval_split = 0.1
            try:
                _original_validate(self)
            finally:
                self.dataset.eval_split = 0.0
        else:
            _original_validate(self)
    finally:
        # 走到 validate 说明 draccus 已解析完临时 yaml，用它做最早的安全删除点
        _cleanup_temp_yaml()


# 应用 monkey-patch：独立 eval 数据集场景下放宽 eval_split 校验
TrainPipelineConfig.validate = _patched_validate


# 临时 yaml（剥离自定义字段后的配置副本）路径，供解析后清理
_TEMP_YAML_PATH: str | None = None


def _cleanup_temp_yaml() -> None:
    """删除临时 yaml（幂等）：draccus 解析完配置后它就没用了，用完立即删除。"""
    global _TEMP_YAML_PATH
    if _TEMP_YAML_PATH is None:
        return
    try:
        Path(_TEMP_YAML_PATH).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"[xvla_train] 清理临时 yaml 失败: {_TEMP_YAML_PATH}: {exc}")
    _TEMP_YAML_PATH = None


def _strip_custom_keys_from_yaml(config_path: str) -> tuple[int | None, dict | None, str]:
    """读取并剥离 yaml 顶层的自定义字段（domain_id / eval_dataset），返回 (domain_id, eval_dataset, tmp_path)。

    draccus 严格匹配 TrainPipelineConfig schema，顶层未知字段会报错，所以先剥离到一份临时 yaml；
    该临时 yaml 在配置解析完成后由 `_cleanup_temp_yaml` 删除，不留残留。
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"yaml 配置文件为空: {config_path}")
    domain_id = cfg.pop("domain_id", None)
    eval_dataset = cfg.pop("eval_dataset", None)
    # 保留原 yaml 的注释块风格：用 default_flow_style=False 让嵌套 dict 仍是 block style
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=str(Path(config_path).parent)
    )
    yaml.safe_dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return domain_id, eval_dataset, tmp.name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="X-VLA 训练入口（带 domain_id 透传）",
        allow_abbrev=False,
    )
    parser.add_argument("--config_path", required=True, help="yaml 配置文件路径")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args, remaining = parser.parse_known_args()

    # 1) 读 yaml 顶层自定义字段（domain_id / eval_dataset），并产出一份 draccus 能消化的临时 yaml
    global _DOMAIN_ID, _EVAL_DATASET_CFG, _TEMP_YAML_PATH
    _DOMAIN_ID, _EVAL_DATASET_CFG, clean_yaml = _strip_custom_keys_from_yaml(args.config_path)
    _TEMP_YAML_PATH = clean_yaml

    if _DOMAIN_ID is None:
        logger.warning(
            "[xvla_train] yaml 顶层未指定 domain_id，将走预训练默认值 0（=Bridge）"
        )
    else:
        logger.info(f"[xvla_train] using domain_id={_DOMAIN_ID}")

    if _EVAL_DATASET_CFG is not None:
        logger.info(f"[xvla_train] using separate eval dataset: {_EVAL_DATASET_CFG}")

    # 2) 重写 sys.argv：让 lerobot_train.train() 用清理后的 yaml + 透传其余 CLI 参数
    sys.argv = ["lerobot-train", "--config_path", clean_yaml, *remaining]

    # 3) 调用 lerobot_train 的 main()，里面会 register plugins + 调 train()
    # train() 被 @parser.wrap() 装饰，会从 sys.argv 解析 TrainPipelineConfig
    try:
        _lt.main()
    finally:
        # 兜底：解析阶段就报错时 validate 不会执行，这里确保临时 yaml 依然被删掉
        _cleanup_temp_yaml()


if __name__ == "__main__":
    main()
