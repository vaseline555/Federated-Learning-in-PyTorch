#!/bin/sh

# FedAvg experiments for LEAF Sent140 dataset
## Sampled 5% of raw dataset as stated in the original paper, as total clients is over 200K...! 
python3 main.py \
    --exp_name FedAvg_LEAF_Sent140 --seed 42 --device cuda \
    --dataset Sent140 \
    --split_type pre --test_size 0.1 \
    --model_name NextCharLSTM --embedding_size 300 --hidden_size 128 --num_layers 2 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 50 --eval_metrics acc1 \
    --R 5000 --E 5 --C 0.0001 --B 10 --beta1 0 \
    --optimizer SGD --lr 0.0003 --lr_decay 1 --lr_decay_step 1 --criterion BCEWithLogitsLoss
