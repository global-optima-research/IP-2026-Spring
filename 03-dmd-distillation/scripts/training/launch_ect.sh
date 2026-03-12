#!/bin/bash
# launch_ect.sh — Launch ECT training in background with nohup

set -euo pipefail

source /data/chenqingzhan/miniconda3/bin/activate fastgen

LOG="/data/chenqingzhan/fastgen_output/ect_train.log"

echo "=== Launching ECT training ==="
echo "Log: $LOG"
echo "Start time: $(date)"

cd /data/chenqingzhan
nohup bash scripts/run_ect_single_gpu.sh > "$LOG" 2>&1 &
PID=$!
echo "PID: $PID"
echo "$PID" > /data/chenqingzhan/fastgen_output/ect_train.pid

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
