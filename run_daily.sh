#!/bin/bash
# IA Monitor L1 每日运行脚本
# 必须在虚拟环境中运行
# cron 示例: 0 17 * * 1-5 cd /path/to/risk_l1 && ./run_daily.sh

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

# 检查虚拟环境
VENV_PYTHON="./.venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at ./.venv" | tee -a logs/cron.log
    echo "Please run: uv venv && uv pip install -e ."
    exit 1
fi

# 运行流水线（使用虚拟环境的 Python）
# 注意：不能用 set -e，否则失败时立即退出，下面的失败日志成为死代码
$VENV_PYTHON src/pipeline.py --config config.yaml >> logs/cron.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Pipeline failed with exit code $EXIT_CODE at $(date)" >> logs/cron.log
fi

exit $EXIT_CODE
