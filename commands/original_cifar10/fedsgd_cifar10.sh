#!/bin/sh

# FedSGD experiments in Figure 4, 9 of (McMahan et al., 2016)
## IID split
python3 main.py \
    --exp_name FedSGD_CIFAR10_CNN_IID --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type iid --test_size -1 \
    --model_name SimpleCNN --crop 24 --randhf 0.5 --randjit 0.5 --imnorm --hidden_size 64 \
    --algorithm fedsgd --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 10000 --C 0.1 --B 0 --beta1 0 \
    --optimizer SGD --lr 0.6 --lr_decay 0.9934 --lr_decay_step 1 --criterion CrossEntropyLoss

## Pathological Non-IID split
python3 main.py \
    --exp_name FedSGD_CIFAR10_CNN_Patho --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type patho --test_size -1 \
    --model_name SimpleCNN --crop 24 --randhf 0.5 --randjit 0.5 --imnorm --hidden_size 64 \
    --algorithm fedsgd --eval_type local --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 10000 --C 0.1 --B 0 --beta1 0 \
    --optimizer SGD --lr 0.45 --lr_decay 0.9934 --lr_decay_step 1 --criterion CrossEntropyLoss
