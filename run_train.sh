#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python HOPE.py \
  --input_file ./datasets/fhad/ \
  --output_file ./checkpoints/fhad/model- \
  --train \
  --val \
  --batch_size 64 \
  --model_def HopeNet \
  --gpu \
  --gpu_number 0 \
  --learning_rate 0.001 \
  --lr_step 100 \
  --lr_step_gamma 0.9 \
  --log_batch 100 \
  --val_epoch 1 \
  --snapshot_epoch 1000 \
  --num_iterations 5000 \
#  --pretrained_model ./checkpoints/obman/model-0.pkl

