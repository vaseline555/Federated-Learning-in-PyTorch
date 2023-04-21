#!/bin/sh

# FedAvg experiments for LEAF Reddit dataset
python3 main.py \
    --exp_name FedAvg_LEAF_Reddit --seed 42 --device cuda \
    --dataset Reddit \
    --split_type pre --test_fraction 0.1 \
    --model_name NextWordLSTM --num_layers 2 --num_embeddings 10000 --embedding_size 256 --hidden_size 256 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 50 --eval_metrics seqacc \
    --R 5000 --E 5 --C 0.013 --B 50 --beta 0 \
    --optimizer SGD --lr 0.0003 --lr_decay 1 --lr_decay_step 1 --criterion Seq2SeqLoss
