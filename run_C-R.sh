#!/bin/bash
gpu_id=$1
date_info="200911"

## C -> R
# mTN
s_name="R"
t_name="C"
echo "--- ${s_name} to ${t_name} | 1"
python train_image.py --gpu_id ${gpu_id} --method RN --num_iteration 20004 --dset office-home  --output_dir RN/${s_name}2${t_name} --source ${s_name} --target ${t_name} --norm_type rn --seed 8346 --run_num rn_offhome_${s_name}2${t_name}_${date_info}_rn

## R -> C
# mTN
s_name="C"
t_name="R"
echo "--- ${s_name} to ${t_name} | 1"
python train_image.py --gpu_id ${gpu_id} --method RN --num_iteration 20004 --dset office-home  --output_dir NO/${s_name}2${t_name} --source ${s_name} --target ${t_name} --norm_type rn --seed 1209 --run_num rn_offhome_${s_name}2${t_name}_${date_info}_rn --dist l2



