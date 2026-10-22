#!/usr/bin/env bash
# ============================================================================
# X-VLA 自定义训练启动脚本（带 domain_id 透传）
# ============================================================================
# 用法：
#   bash run_xvla_train.sh                              # 使用默认 xvla_demo_config.yaml
#   bash run_xvla_train.sh --config_path ./xxx.yaml     # 切换其他 yaml
#   bash run_xvla_train.sh --steps=500 --batch_size=2   # 追加/覆盖任意 CLI 参数
#
# 说明：CLI 参数优先级高于 yaml，因此可在不改配置文件的情况下临时覆盖。
# domain_id 由 yaml 顶层的 `domain_id` 字段控制，会被 xvla_train.py 自动透传
# 到 preprocessor_overrides["xvla_add_domain_id"]，影响 soft_prompt_hub 查表。

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# SwanLab / WandB：日志后端。无外网时把 yaml 里的 wandb.mode 改为 offline
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-lxinhZyaQbBOWbcnFbfdX}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---------------------------------------------------------------------------
# 配置默认值
# ---------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-./xvla_demo_config.yaml}"

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
echo "=========================================="
echo " X-VLA training with domain_id injection"
echo " config: ${CONFIG_PATH}"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " extra args: $*"
echo "=========================================="

python "${SCRIPT_DIR}/xvla_train.py" \
    --config_path "${CONFIG_PATH}" \
    --wandb.backend=swanlab \
    "$@"
