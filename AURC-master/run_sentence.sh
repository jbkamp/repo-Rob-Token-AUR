#!/usr/bin/sh

python3 src/run_AURC_sentence.py \
    --card_number=0 \
    --train \
    --eval \
    --target_domain='In-Domain' \
    --epochs=10 \
    --save_prediction \
