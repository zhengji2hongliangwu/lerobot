#!/usr/bin/env bash
# ============================================================================
# X-VLA 训练启动脚本（小数据集 lerobot/pusht 跑通示例）
# ============================================================================
# 用法：
#   bash lerobot_xvla_train.sh                     # 使用默认 yaml 配置
#   bash lerobot_xvla_train.sh --steps=500         # 追加/覆盖任意 CLI 参数
#
# 说明：CLI 参数优先级高于 yaml，因此可在不改配置文件的情况下临时覆盖。

# 若被 sh(dash) 执行，自动改用 bash 重新执行自身
# （dash 不支持 pipefail，且 set 遇非法选项会直接终止脚本）
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# 脚本所在目录（保证相对路径 ./xvla_pusht_config.yaml 可用）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 环境准备
# ---------------------------------------------------------------------------
# 激活 conda 的 lerobot 环境（lerobot-train 安装在该环境，未激活会报 command not found）
# shellcheck disable=SC1091
source /home/jiz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# HuggingFace 镜像（国内加速；huggingface.co 不可直连）
export HF_ENDPOINT=https://hf-mirror.com
# hf-mirror 不支持 xet 传输协议，必须关闭，否则报 401 Unauthorized
export HF_HUB_DISABLE_XET=1

# 指定使用的 GPU（按需修改；当前机器 GPU 0 被占用，1/2/3 空闲）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# # WandB：无外网或未登录时用 offline 模式，日志落在 output_dir/wandb 下
# export WANDB_MODE="${WANDB_MODE:-online}"
# # 如需在线同步：先在 https://wandb.ai/authorize 拿到 40 位以上的 key，
# # 再取消下面注释填入（key 非法会导致 wandb.init 直接抛 AuthenticationError 终止训练）：
# export WANDB_API_KEY=lxinhZyaQbBOWbcnFbfdX

# SwanLab：日志后端已在下面用 --wandb.backend=swanlab 指定
# 无外网/未登录时，把 xvla_pusht_config.yaml 里的 wandb.mode 改为 offline，
# 日志落在 output_dir/swanlog 下；在线同步时 mode 用 online（映射为 swanlab cloud）。
# API key 在 https://swanlab.cn 获取；key 非法会导致 swanlab.init 报错终止训练。
# 注意：需先安装 swanlab（pip install swanlab）。
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-lxinhZyaQbBOWbcnFbfdX}"

# 可选：完全离线（权重/数据集已缓存）时开启，避免联网校验
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# 限制线程数，避免 DataLoader worker 与主进程争抢 CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
echo "=========================================="
echo " X-VLA training on lerobot/pusht"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " logging backend: swanlab"
echo " SWANLAB_API_KEY=${SWANLAB_API_KEY:+***set***}"
echo "=========================================="

lerobot-train \
    --config_path ./xvla_test_config.yaml \
    --wandb.backend=swanlab \
    "$@"
