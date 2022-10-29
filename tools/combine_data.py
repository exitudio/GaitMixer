import numpy as np
import sys
sys.path.append('../src')
import os
from tqdm import tqdm
from datasets.gait import CasiaBPose

train_dataset = CasiaBPose(
    '../data/casia-b_pose_train_valid.csv',
    sequence_length=60,
)
test_dataset = CasiaBPose('../data/casia-b_pose_test.csv',
    sequence_length=60)

os.makedirs('../data/casiab_npy', exist_ok=True)
def save_npy(dataset):
    for data, target in tqdm(dataset):
        np.save('../data/casiab_npy/'+'_'.join(map(str,target))+'.npy', data)
save_npy(test_dataset)
save_npy(train_dataset)