#!/bin/sh

# FedAvg experiments for LEAF Shakespeare dataset
python3 main.py \
    --exp_name FedAvg_LEAF_Shakespeare --seed 42 --device cuda \
    --dataset Shakespeare \
    --split_type pre --test_size 0.1 \
    --model_name NextCharLSTM --num_embeddings 80 --embedding_size 8 --hidden_size 256 --num_layers 2 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 50 --eval_metrics acc1 acc5 \
    --R 5000 --E 5 --C 0.016 --B 10 --beta1 0 \
    --optimizer SGD --lr 0.0003 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss
