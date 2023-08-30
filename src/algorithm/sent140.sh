#!/bin/sh
## 2,460 clients, 2 classes

for s in 42 1023 59999
do
    for a in 0 0.1 1 10
    do
        python3 main.py \
        --exp_name "Sent140_FedAvg_Fixed_${a} (${s})" --seed $s --device cuda:0 \
        --dataset Sent140 --learner fixed --alpha $a \
        --split_type pre --rawsmpl 0.01 --test_size 0.2 \
        --model_name Sent140LSTM --embedding_size 300 --hidden_size 256 --num_layers 2 \
        --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 5000 --eval_metrics acc1 \
        --R 500 --C 0.0021 --E 5 --B 10 \
        --optimizer SGD --lr 0.3 --lr_decay 0.98 --lr_decay_step 1 --criterion BCEWithLogitsLoss &

        python3 main.py \
        --exp_name "Sent140_FedAvg_AdaHedge_${a} (${s})" --seed $s --device cuda:0 \
        --dataset Sent140 --learner ah --alpha $a \
        --split_type pre --rawsmpl 0.01 --test_size 0.2 \
        --model_name Sent140LSTM --embedding_size 300 --hidden_size 256 --num_layers 2 \
        --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 5000 --eval_metrics acc1 \
        --R 500 --C 0.0021 --E 5 --B 10 \
        --optimizer SGD --lr 0.3 --lr_decay 0.98 --lr_decay_step 1 --criterion BCEWithLogitsLoss &

        python3 main.py \
        --exp_name "Sent140_FedAvg_SoftBayes_${a} (${s})" --seed $s --device cuda:0 \
        --dataset Sent140 --learner sb --alpha $a \
        --split_type pre --rawsmpl 0.01 --test_size 0.2 \
        --model_name Sent140LSTM --embedding_size 300 --hidden_size 256 --num_layers 2 \
        --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 100 --eval_metrics acc1 \
        --R 500 --C 0.0021 --E 5 --B 10 \
        --optimizer SGD --lr 0.3 --lr_decay 0.98 --lr_decay_step 1 --criterion BCEWithLogitsLoss
    done
    wait
done

python3 main.py \
        --exp_name Sent140_FedAvg_Fixed --seed 42 --device cuda:2 \
        --dataset Sent140 --learner fixed \
        --split_type pre --rawsmpl 0.01 --test_size 0.2 \
        --model_name Sent140LSTM --embedding_size 300 --hidden_size 80 --num_layers 2 \
        --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 1000 --eval_metrics acc1 \
        --R 1000 --C 0.0021 --E 5 --B 10 \
        --optimizer SGD --lr 0.0003 --lr_decay 0.9995 --lr_decay_step 1 --criterion BCEWithLogitsLoss
        
