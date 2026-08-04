#!/bin/bash

for seed in 1 2 3 4 5
do
  python run.py \
    --seed "$seed" \
    --dataset ushcn \
    --history 24 \
    --batch_size 192 \
    --ctx_len 512 \
    --fcast_len 205 \
    --use_padding periodic \
    --num_hops 1 \
    --backbone MOMENT \
    --model_name TREAD \
    --gpu 0
done
