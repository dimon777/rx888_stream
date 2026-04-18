#!/bin/bash

# Check if frequency is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <frequency_in_MHz>"
  echo "Example: $0 134.8"
  exit 1
fi

# Convert frequency to Hz
BASE_FREQ_MHZ=$1
FREQUENCY=$(echo "$BASE_FREQ_MHZ * 1000000" | bc | cut -d. -f1)

# Offset tuning by 50 kHz to avoid the DC center spike
OFFSET=50000
TUNER_FREQ=$((FREQUENCY + OFFSET))
# Shift ratio = -OFFSET / SAMPLE_RATE = -50000 / 10000000 = -0.005
SHIFT_RATIO="-0.005"

SAMPLE_RATE=10000000
DECIM=200
# Output rate after decimation is 50,000 Hz

echo "Tuning to $BASE_FREQ_MHZ MHz (Hardware offset to $(($TUNER_FREQ/1000000)) MHz)"

./target/release/rx888_stream vhf \
    -f SDDC_FX3.img \
    --frequency "$TUNER_FREQ" \
    --sample-rate "$SAMPLE_RATE" \
    -g 100 \
    --vhf-lna 20 \
    --vhf-vga 12 \
    -o - \
| csdr convert_s16_f \
| csdr shift_addition_fc "$SHIFT_RATIO" \
| csdr fastddc_fwd_cc "$DECIM" \
| csdr squelch_and_smeter_cc -30 1 \
| csdr amdemod_cf \
| csdr fastdcblock_ff \
| csdr lowpass_fir_fft_ff 0.08 200 \
| csdr agc_ff \
| csdr gain_ff 3 \
| csdr rational_resampler_ff 24 25 \
| csdr convert_f_s16 \
| aplay -r 48000 -f S16_LE -t raw -c 1
