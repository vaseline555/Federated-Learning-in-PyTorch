#!/bin/sh

# FedAvg experiments for LEAF CelebA dataset
python3 main.py \
    --exp_name FedAvg_LEAF_CelebA --seed 42 --device cuda \
    --dataset CelebA \
    --split_type pre --test_size 0.1 \
    --model_name TwoCNN --resize 84 --hidden_size 32 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 50 --eval_metrics acc1 \
    --R 5000 --E 5 --C 0.001 --B 10 --beta1 0 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 1 --criterion BCEWithLogitsLoss
