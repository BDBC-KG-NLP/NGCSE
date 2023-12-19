#!/bin/bash

git lfs install
git clone https://huggingface.co/datasets/H34lthy/NGCSE_hybrid_dataset
mv NGCSE_hybrid_dataset/*.jsonl ./
mv NGCSE_hybrid_dataset/data_domain ./
rm -rf NGCSE_hybrid_dataset