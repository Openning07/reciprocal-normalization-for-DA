#!/bin/bash
gpu_id=$1
date_info="200911"

## A -> C
# mTN
s_name="train"
t_name="val"
echo "--- ${s_name} to ${t_name} | 1"
python test_image.py --gpu_id ${gpu_id} --method RN --net ResNet50 --dset visda  --seed 2021 




