#!/bin/sh

# FedSGD experiments in Table 2, Figure 2, 3 of (McMahan et al., 2016)
## Note: this is equivalent to Shakespeare dataset under LEAF benchmark
## Role & Play Non-IID split
python3 main.py \
    --exp_name FedSGD_Shakespeare_NextCharLSTM --seed 42 --device cuda \
    --dataset Shakespeare \
    --split_type pre --test_size 0.2 \
    --model_name NextCharLSTM --num_embeddings 80 --embedding_size 8 --hidden_size 256 --num_layers 2 \
    --algorithm fedsgd --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics acc1 acc5 \
    --R 5000 --C 0.1 --B 0 --beta 0 \
    --optimizer SGD --lr 3.0 --lr_decay 0.9999 --lr_decay_step 25 --criterion CrossEntropyLoss
