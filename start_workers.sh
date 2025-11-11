#!/bin/bash

# start_workers_safe.sh

NUM_WORKERS=4
START_PORT=8030
BASE_DIR="worker"
TIMEOUT=10

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num) NUM_WORKERS="$2"; shift 2 ;;
        -p|--port) START_PORT="$2"; shift 2 ;;
        -d|--dir) BASE_DIR="$2"; shift 2 ;;
        -t|--timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-n NUM] [-p PORT] [-t TIMEOUT]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

PID_FILE=".worker_pids"
> "$PID_FILE"

echo "Starting $NUM_WORKERS workers (waiting for each to be ready)..."
echo ""

for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((START_PORT + i))
    LOG_FILE="worker_$PORT.log"
    
    echo "[$((i+1))/$NUM_WORKERS] Starting worker on localhost:$PORT..."
    
    # Start worker
    go run "$BASE_DIR/worker.go" -ip="localhost:$PORT" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" >> "$PID_FILE"
    
    # Wait for worker to be ready
    ELAPSED=0
    READY=false
    
    while [ $ELAPSED -lt $TIMEOUT ]; do
        # Check if worker logged that it's listening
        if grep -q "Listening on port" "$LOG_FILE" 2>/dev/null; then
            echo "    ✓ Worker ready (took ${ELAPSED}s)"
            READY=true
            break
        fi
        
        # Check if worker logged broker registration
        if grep -q "Dialed broker" "$LOG_FILE" 2>/dev/null; then
            echo "    ✓ Worker registered with broker (took ${ELAPSED}s)"
            READY=true
            break
        fi
        
        # Check if process died
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "    ✗ Worker died! Check $LOG_FILE"
            break
        fi
        
        sleep 0.5
        ELAPSED=$((ELAPSED + 1))
    done
    
    if [ "$READY" = false ]; then
        echo "    ⚠ Worker didn't confirm ready within ${TIMEOUT}s"
        echo "    Check $LOG_FILE for errors"
    fi
    
    # Small additional delay for broker processing
    sleep 0.5
done

echo ""
echo "All $NUM_WORKERS workers started!"
echo "Logs: worker_*.log"
