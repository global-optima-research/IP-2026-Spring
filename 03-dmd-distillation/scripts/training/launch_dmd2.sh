#!/bin/bash
# launch_dmd2.sh — Launch DMD2 training in background with nohup
# Run on server: bash launch_dmd2.sh

set -euo pipefail

source /data/chenqingzhan/miniconda3/bin/activate fastgen

LOG="/data/chenqingzhan/fastgen_output/dmd2_train.log"

echo "=== Launching DMD2 training ==="
echo "Log: $LOG"
echo "Start time: $(date)"

cd /data/chenqingzhan
nohup bash scripts/run_dmd2_single_gpu.sh > "$LOG" 2>&1 &
PID=$!
echo "PID: $PID"
echo "$PID" > /data/chenqingzhan/fastgen_output/dmd2_train.pid

sleep 3

echo ""
echo "=== Process check ==="
if ps -p $PID > /dev/null 2>&1; then
    echo "Training process is running (PID=$PID)"
else
    echo "WARNING: Process not found!"
fi

echo ""
echo "=== Initial log output ==="
tail -20 "$LOG" 2>/dev/null || echo "Log not yet written"

echo ""
echo "Monitor with: tail -f $LOG"
echo "Check GPU with: nvidia-smi"
