#!/bin/bash

python HOPE.py \
  --input_file ./datasets/obman_normal/ \
  --test \
  --batch_size 2048 \
  --model_def GraphUNet \
  --pretrained_model ./checkpoints/normal/model-1000.pkl \
  --gpu \
  --gpu_number 0 1

