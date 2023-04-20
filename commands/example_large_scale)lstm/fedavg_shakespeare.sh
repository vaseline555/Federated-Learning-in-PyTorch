#!/bin/sh

# FedAvg experiments in Figure 5, 10 of (McMahan et al., 2016)
## Note: this is equivalent to Reddit dataset under LEAF benchmark
## Role & Play Non-IID split
python3 main.py \
    --exp_name FedAvg_Shakespeare_NextCharLSTM --seed 42 --device cuda \
    --dataset Shakespeare \
    --split_type pre --test_fraction 0.2 \
    --model_name NextCharLSTM --num_embeddings 80 --embedding_size 8 --hidden_size 256 \
    --algorithm fedavg --eval_type local --eval_every 1 \
    --R 5000 --E 5 --C 0.1 --B 10 --beta 0 \
    --optimizer SGD --lr 1.47 --lr_decay 0.9999 --lr_decay_step 50 --criterion CrossEntropyLoss
