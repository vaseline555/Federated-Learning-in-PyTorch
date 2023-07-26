#!/bin/sh

# FedProx experiments 
## IID split
python3 main.py \
    --exp_name FedProx_CIFAR10_CNN_IID --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type iid --test_size -1 \
    --model_name SimpleCNN --crop 24 --randhf 0.1 --randjit 0.1 --hidden_size 64 \
    --algorithm fedprox --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 5000 --E 5 --C 0.1 --B 50 --beta 0 \
    --optimizer SGD --lr 0.17 --lr_decay 0.99 --lr_decay_step 1 --criterion CrossEntropyLoss \
    --mu 0.01

## Pathological Non-IID split
python3 main.py \
    --exp_name FedProx_CIFAR10_CNN_Patho --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type patho --test_size -1 \
    --model_name SimpleCNN --crop 24 --randhf 0.1 --randjit 0.1 --hidden_size 64 \
    --algorithm fedprox --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 5000 --E 5 --C 0.1 --B 50 --beta 0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.9999 --lr_decay_step 1 --criterion CrossEntropyLoss \
    --mu 0.01
