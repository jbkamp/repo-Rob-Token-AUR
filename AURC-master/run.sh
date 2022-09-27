#!/usr/bin/sh

python3 src/run_AURC_token.py \
    --card_number=0 \
    --train \
    --eval \
    --crf \
    --target_domain='In-Domain' \
    --save_prediction \
    --epochs=10 \
