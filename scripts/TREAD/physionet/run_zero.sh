#!/bin/bash

for seed in 1 2 3 4 5
do
  python run.py \
    --seed "$seed" \
    --dataset physionet \
    --history 24 \
    --batch_size 32 \
    --ctx_len 128 \
    --fcast_len 128 \
    --use_padding zero \
    --num_hops 3 \
    --backbone MOMENT \
    --model_name TREAD \
    --gpu 0
done
