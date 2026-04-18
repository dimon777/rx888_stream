./target/release/rx888_stream vhf -f SDDC_FX3.img \
  --frequency 134800000 --sample-rate 10000000 \
  -g 127 --vhf-lna 29 --vhf-vga 15 \
  -o - 2>/dev/null \
| head -c 10000 | od -s | head -3

# Valid ranges:
# 0000000  28046 -22251   6806  -2636   -501 -17411 -11194   2596
# 0000020 -18933   7307 -29850   9252   6610 -10230  21125 -21149
# 0000040 -32768  27753  26083    492  -1641   6123   8657   9946
