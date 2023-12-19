#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --model_name_or_path $1 \
  --model_type bert \
  --task_set sts \
  --mode test
