#!/usr/bin/env bash
# ============================================================================
# X-VLA 训练启动脚本（domain_id 透传 + 保存 eval_loss 最低的模型）
# ============================================================================
# 用法：
#   bash run_xvla_train_best.sh                                 # 默认 xvla_demo_config_best.yaml
#   bash run_xvla_train_best.sh --config_path ./xxx.yaml        # 切换其他 yaml
#   bash run_xvla_train_best.sh --steps=500 --batch_size=2      # 追加/覆盖任意 CLI 参数
#
# 说明：
#   · 入口是 xvla_train_best.py —— 在原训练入口基础上，额外把 eval_loss 最低的 step
#     标记为 checkpoints/best（软链接，与官方 `last` 同款），并写 best_model.json。
#   · CLI 参数优先级高于 yaml，可在不改配置文件的情况下临时覆盖。
#   · domain_id 由 yaml 顶层的 `domain_id` 字段控制，会被自动透传到
#     preprocessor_overrides["xvla_add_domain_id"], 影响 soft_prompt_hub 查表。
#
# 训练结束后用最优模型评测/部署：
#   --policy.path=<output_dir>/checkpoints/best/pretrained_model

# 若被 sh(dash) 执行，自动改用 bash 重新执行自身
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 环境准备
# ---------------------------------------------------------------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 关键：facebook/bart-large tokenizer 与 lerobot/xvla-base 权重均已在本地 HF 缓存
# （~/.cache/huggingface/hub/models--facebook--bart-large、models--lerobot--xvla-base），
# 但 TokenizerProcessorStep 每次启动都会联网（hf-mirror）校验文件，网络抖动时报
# "[Errno 110] Connection timed out" 使预处理器加载失败。强制离线直接用本地缓存即可，
# 彻底避开联网校验。若之后需要下载新的权重/数据集，注释掉下面两行即可恢复联网。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# SwanLab / WandB：日志后端。无外网时把 yaml 里的 wandb.mode 改为 offline
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-lxinhZyaQbBOWbcnFbfdX}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---------------------------------------------------------------------------
# 配置默认值
# ---------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-./xvla_demo_config_best.yaml}"

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
echo "=========================================="
echo " X-VLA training (save best-by-eval checkpoint)"
echo " config: ${CONFIG_PATH}"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " extra args: $*"
echo "=========================================="

python "${SCRIPT_DIR}/xvla_train_best.py" \
    --config_path "${CONFIG_PATH}" \
    --wandb.backend=swanlab \
    "$@"
