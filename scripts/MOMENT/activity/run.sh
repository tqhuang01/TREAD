#!/bin/bash

for seed in 1 2 3 4 5
do
  python run.py \
    --seed "$seed" \
    --dataset activity \
    --history 3000 \
    --batch_size 32 \
    --ctx_len 256 \
    --fcast_len 98 \
    --use_padding zero \
    --backbone MOMENT \
    --model_name FM_ZP \
    --gpu 0
done
