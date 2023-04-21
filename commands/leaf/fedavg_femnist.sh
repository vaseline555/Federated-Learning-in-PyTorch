#!/bin/sh

# FedAvg experiments for LEAF FEMNIST dataset
python3 main.py \
    --exp_name FedAvg_LEAF_FEMNIST --seed 42 --device cuda \
    --dataset FEMNIST \
    --split_type pre --test_fraction 0.1 \
    --model_name TwoCNN --resize 28 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 50 --eval_metrics acc1 acc5 \
    --R 5000 --E 5 --C 0.003 --B 10 --beta 0 \
    --optimizer SGD --lr 0.0003 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss
