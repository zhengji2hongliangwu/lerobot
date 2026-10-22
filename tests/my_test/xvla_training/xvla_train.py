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
#   4. 重写 sys.argv 让 lerobot_train.train() 走标准流程，透传其余 CLI 参数
#
# 不动 lerobot 源码即可让 soft_prompt_hub 查表取到正确的 domain 行。
# ============================================================================

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import yaml

# 必须在调用 train() 之前 import + patch
import lerobot.scripts.lerobot_train as _lt

logger = logging.getLogger("xvla_train")

# 全局 domain_id，由 monkey-patch 闭包捕获
_DOMAIN_ID: int | None = None

# 临时 yaml（剥离 domain_id 后的配置副本）路径，供解析后清理
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


def _strip_domain_id_from_yaml(config_path: str) -> tuple[int | None, str]:
    """Read top-level `domain_id` from yaml, write a tmp yaml without it, return (id, tmp_path).

    draccus 严格匹配 TrainPipelineConfig schema，顶层未知字段会报错，所以先剥离。
    该临时 yaml 在配置解析完成后由 `_cleanup_temp_yaml` 删除，不留残留。
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"yaml 配置文件为空: {config_path}")
    domain_id = cfg.pop("domain_id", None)
    # 保留原 yaml 的注释块风格：用 default_flow_style=False 让嵌套 dict 仍是 block style
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=str(Path(config_path).parent)
    )
    yaml.safe_dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return domain_id, tmp.name


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

    # 1) 读 yaml 顶层 domain_id，并产出一份 draccus 能消化的临时 yaml
    global _DOMAIN_ID, _TEMP_YAML_PATH
    _DOMAIN_ID, clean_yaml = _strip_domain_id_from_yaml(args.config_path)
    _TEMP_YAML_PATH = clean_yaml

    if _DOMAIN_ID is None:
        logger.warning(
            "[xvla_train] yaml 顶层未指定 domain_id，将走预训练默认值 0（=Bridge）"
        )
    else:
        logger.info(f"[xvla_train] using domain_id={_DOMAIN_ID}")

    # 2) 重写 sys.argv：让 lerobot_train.train() 用清理后的 yaml + 透传其余 CLI 参数
    sys.argv = ["lerobot-train", "--config_path", clean_yaml, *remaining]

    # 3) 调用 lerobot_train 的 main()，里面会 register plugins + 调 train()
    # train() 被 @parser.wrap() 装饰，会从 sys.argv 解析 TrainPipelineConfig
    try:
        _lt.main()
    finally:
        # 解析/训练结束后立即删除临时 yaml，避免残留
        _cleanup_temp_yaml()


if __name__ == "__main__":
    main()
