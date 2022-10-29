#!/bin/bash

conda activate GaitSelfFormer
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train.py casia-b-query /home/epinyoan/git/GaitSelfFormer/v2_all/data/casiab_npy \
                --train_id 1-24 \
                --test_id 25-124 \
                --sampler_num_sample 8