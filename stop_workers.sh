#!/bin/bash

# stop_workers.sh - Stop all workers

PID_FILE=".worker_pids"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Are workers running?"
    exit 1
fi

echo "Stopping workers..."

while read -r PID; do
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Killing worker PID $PID"
        kill "$PID" 2>/dev/null
    fi
done < "$PID_FILE"

# Wait a moment, then force kill if needed
sleep 1

while read -r PID; do
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Force killing worker PID $PID"
        kill -9 "$PID" 2>/dev/null
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "All workers stopped."
