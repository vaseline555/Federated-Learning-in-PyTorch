#!/bin/sh

# FedSGD experiments in Figure 4, 9 of (McMahan et al., 2016)
## IID split
python3 main.py \
    --exp_name FedSGD_CIFAR10_CNN_IID --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type iid --test_fraction 0 \
    --model_name TwoCNN --resize 24 --randhf 0.5 --randjit 0.5 --hidden_size 32 \
    --algorithm fedsgd --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 5000 --C 0.1 --B 0 --beta 0 \
    --optimizer SGD --lr 0.45 --lr_decay 0.9934 --lr_decay_step 1 --criterion CrossEntropyLoss

## Pathological Non-IID split
python3 main.py \
    --exp_name FedSGD_CIFAR10_CNN_Patho --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type patho --test_fraction 0 \
    --model_name TwoCNN --resize 24 --randhf 0.5 --randjit 0.5 --hidden_size 32 \
    --algorithm fedsgd --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 5000 --C 0.1 --B 0 --beta 0 \
    --optimizer SGD --lr 0.15 --lr_decay 0.9999 --lr_decay_step 5 --criterion CrossEntropyLoss
