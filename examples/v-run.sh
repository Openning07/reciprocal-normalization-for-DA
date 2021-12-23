#!/bin/bash
gpu_id=$1
date_info="200911"

## A -> C
# mTN

## C -> A
# mTN
s_name="train"
t_name="val"
echo "--- ${s_name} to ${t_name} | 1"
python train_image.py --gpu_id ${gpu_id} --net ResNet50  --method RN --num_iteration 20004 --dset visda --output_dir RN/${s_name}2${t_name} --source ${s_name} --target ${t_name} --norm_type rn --seed 2021 --run_num rn_offhome_${s_name}2${t_name}_${date_info}_rn_50_visda 



