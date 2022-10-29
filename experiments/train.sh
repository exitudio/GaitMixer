#!/bin/bash

conda activate GaitMixer
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py casia-b ../data/casia-b_pose_train_valid.csv --valid_data_path ../data/casia-b_pose_test.csv