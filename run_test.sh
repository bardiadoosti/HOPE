#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python HOPE.py \
  --input_file ./datasets/fhad/ \
  --test \
  --batch_size 64 \
  --model_def HopeNet \
  --gpu \
  --gpu_number 0 \
  --pretrained_model ./checkpoints/fhad/model-5000.pkl

