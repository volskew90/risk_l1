#!/bin/bash
# IA Monitor L1 每日运行脚本
# cron 示例: 0 17 * * 1-5 cd /path/to/risk_l1 && ./run_daily.sh

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

# 优先使用 uv run python 执行，若未找到 uv 则回退至 ./.venv/bin/python3
if command -v uv >/dev/null 2>&1; then
    uv run python src/pipeline.py --config config.yaml >> logs/cron.log 2>&1
    EXIT_CODE=$?
elif [ -f "./.venv/bin/python3" ]; then
    ./.venv/bin/python3 src/pipeline.py --config config.yaml >> logs/cron.log 2>&1
    EXIT_CODE=$?
else
    echo "Error: Neither uv nor ./.venv/bin/python3 found" | tee -a logs/cron.log
    exit 1
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "Pipeline failed with exit code $EXIT_CODE at $(date)" >> logs/cron.log
fi

exit $EXIT_CODE
