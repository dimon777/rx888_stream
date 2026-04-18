#!/bin/bash

START=118.000
END=137.000

WORKERS=3

RANGE=$(echo "$END - $START" | bc)
CHUNK=$(echo "$RANGE / $WORKERS" | bc -l)

echo "[*] Starting $WORKERS RX888 workers..."

for i in $(seq 0 $((WORKERS - 1))); do

    WSTART=$(awk "BEGIN {printf \"%.3f\", $START + ($i * $CHUNK)}")
    WEND=$(awk "BEGIN {printf \"%.3f\", $START + (($i + 1) * $CHUNK)}")

    echo "[*] Worker $i: $WSTART → $WEND"

    ./rx888_worker.sh "$WSTART" "$WEND" &
done

wait