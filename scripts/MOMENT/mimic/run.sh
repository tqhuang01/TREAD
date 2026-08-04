#!/bin/bash

for seed in 1 2 3 4 5
do
  python run_distributed.py \
    --seed "$seed" \
    --dataset mimic \
    --history 24 \
    --batch_size 32 \
    --ctx_len 280 \
    --fcast_len 414 \
    --use_padding zero \
    --backbone MOMENT \
    --model_name FM_ZP
done
