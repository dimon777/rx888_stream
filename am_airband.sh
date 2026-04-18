#!/bin/bash

# Check if frequency is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <frequency_in_MHz>"
  echo "Example: $0 134.8    (Airband)"
  echo "Example: $0 121.5    (Emergency)"
  exit 1
fi

# Convert frequency to Hz
FREQUENCY=$(echo "$1 * 1000000" | bc | cut -d. -f1)

# Sample rate: 10 MHz for RX-888 VHF
SAMPLE_RATE=10000000
DECIM=200
# Output rate after decimation: 10,000,000 / 200 = 50,000 Hz
# Resample 50,000 -> 48,000 Hz:  50000 * 24/25 = 48000

echo "Tuning to $1 MHz ($FREQUENCY Hz)"

./target/release/rx888_stream vhf \
    -f SDDC_FX3.img \
    --frequency "$FREQUENCY" \
    --sample-rate "$SAMPLE_RATE" \
    -g 127 \
    --vhf-lna 29 \
    --vhf-vga 15 \
    -o - \
| csdr convert_s16_f \
| csdr shift_addition_fc 0 \
| csdr fastddc_fwd_cc "$DECIM" \
| csdr amdemod_cf \
| csdr fastdcblock_ff \
| csdr agc_ff \
| csdr gain_ff 3 \
| csdr rational_resampler_ff 24 25 \
| csdr convert_f_s16 \
| aplay -r 48000 -f S16_LE -t raw -c 1