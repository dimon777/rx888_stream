#!/bin/bash

START=$1
END=$2
STEP=0.025

SAMPLE_RATE=10000000
DECIM=200
OFFSET=50000

THRESHOLD=0.015

RXBIN="./target/release/rx888_stream"
FIRMWARE="-f SDDC_FX3.img"

mhz_to_hz() {
    awk "BEGIN {printf \"%d\", $1 * 1000000}"
}

measure_power() {
    local freq_mhz=$1
    local freq_hz=$(mhz_to_hz "$freq_mhz")
    local tuner_freq=$((freq_hz + OFFSET))

    $RXBIN vhf \
        --frequency "$tuner_freq" \
        --sample-rate "$SAMPLE_RATE" \
        -g 100 \
        --vhf-lna 20 \
        --vhf-vga 12 \
        -o - \
    | csdr convert_s16_f \
    | csdr shift_addition_fc -0.005 \
    | csdr fastddc_fwd_cc "$DECIM" \
    | csdr amdemod_cf \
    | csdr fastdcblock_ff \
    | csdr agc_ff \
    | csdr convert_f_s16 \
    | od -An -t f4 \
    | tail -n 1
}

FREQ=$START

while true; do
    while (( $(echo "$FREQ <= $END" | bc -l) )); do

        POWER=$(measure_power "$FREQ")

        if awk "BEGIN {exit !($POWER > $THRESHOLD)}"; then
            echo "[DETECT] ${FREQ} MHz power=$POWER"
        fi

        FREQ=$(awk "BEGIN {printf \"%.3f\", $FREQ + $STEP}")
    done

    FREQ=$START
done