#!/bin/bash

for seed in 1 2 3 4 5
do
  for padding in zero periodic none
  do
    python run.py --seed "$seed" --dataset physionet --history 24 --batch_size 32 --ctx_len 128 --fcast_len 128 --use_padding "$padding" --num_hops 3 --backbone MOMENT --model_name TREAD --gpu 0
    python run_distributed.py --seed "$seed" --dataset mimic --history 24 --batch_size 8 --ctx_len 280 --fcast_len 414 --use_padding "$padding" --num_hops 3 --backbone MOMENT --model_name TREAD
    python run.py --seed "$seed" --dataset activity --history 3000 --batch_size 32 --ctx_len 256 --fcast_len 98 --use_padding "$padding" --num_hops 3 --backbone MOMENT --model_name TREAD --gpu 0
    python run.py --seed "$seed" --dataset ushcn --history 24 --batch_size 192 --ctx_len 512 --fcast_len 205 --use_padding "$padding" --num_hops 1 --backbone MOMENT --model_name TREAD --gpu 0
  done
done
