#!/bin/bash

mkdir -p results/runs

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_sts.jsonl \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --save_steps 125 \
    --logging_steps 125 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --use_native_loss \
    --use_loss3 \
    --temp 5e-2 \
    --margins '[[5e-3, 10e-2], []]' \
    --beta 1 \
    --pooler_type mask \
    --num_layers 12 \
    --feature_size 768 \
    --model_type bert \
    --output_dir results/runs/bert_nli_sts \
    --overwrite_output_dir \
    --do_train \
    --do_eval \