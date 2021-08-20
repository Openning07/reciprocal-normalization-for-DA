#!/bin/bash
gpu_id=$1
date_info="200911"

## A -> P
# mTN
s_dset_path="${art_txt}"
t_dset_path="${clipart_txt}"
s_name="A"
t_name="P"
echo "--- ${s_name} to ${t_name} | 1"
python train_image.py --gpu_id ${gpu_id} --method RN --num_iteration 20004 --dset office-home  --output_dir RN/${s_name}2${t_name} --source ${s_name} --target ${t_name} --norm_type rn --seed 6659 --run_num rn_offhome_${s_name}2${t_name}_${date_info}_rn 

## P -> A
# mTN
s_dset_path="${clipart_txt}"
t_dset_path="${art_txt}"
s_name="P"
t_name="A"
echo "--- ${s_name} to ${t_name} | 1"
python train_image.py --gpu_id ${gpu_id} --method RN --num_iteration 20004 --dset office-home  --output_dir NO/${s_name}2${t_name} --source ${s_name} --target ${t_name} --norm_type rn --seed 4556 --run_num rn_offhome_${s_name}2${t_name}_${date_info}_rn 



