#!/bin/sh

# FedProx experiments
## IID split
for b in 0 10
do
    for c in 0.0 0.1 0.2 0.5 1.0
    do
        python3 main.py \
            --exp_name "FedProx_MNIST_CNN_IID_C${c}_B${b}" --seed 42 --device cuda \
            --dataset MNIST \
            --split_type iid --test_size 0 \
            --model_name TwoCNN --resize 28 --hidden_size 200 \
            --algorithm fedprox --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
            --K 100 --R 1000 --E 5 --C $c --B $b --beta 0 \
            --optimizer SGD --lr 0.215 --lr_decay 0.95 --lr_decay_step 25 --criterion CrossEntropyLoss \
            --mu 0.01 
    done     
done 

## Pathological Non-IID split
for b in 0 10
do
    for c in 0.0 0.1 0.2 0.5 1.0
    do
        python3 main.py \
            --exp_name "FedProx_MNIST_CNN_Patho_C${c}_B${b}" --seed 42 --device cuda \
            --dataset MNIST \
            --split_type patho --test_size 0 \
            --model_name TwoNNC --resize 28 --hidden_size 200 \
            --algorithm fedprox --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
            --K 100 --R 1000 --E 1 5-C $c --B $b --beta 0 \
            --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_decay_step 10 --criterion CrossEntropyLoss \
            --mu 0.01
    done     
done 
