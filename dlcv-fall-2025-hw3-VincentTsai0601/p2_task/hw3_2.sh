#!/bin/bash
python3 "$(dirname "$0")/inference.py" \
    --images_root "$1" \
    --output "$2" \
    --decoder_path "$3"

