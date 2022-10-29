# Training
Basic train:
```shell
python train.py casia-b ../data/casia-b_pose_train_valid.csv --valid_data_path ../data/casia-b_pose_test.csv
```

Use ```--weight_path``` to load pretrain model.

## Separate Train & Test by Subject ID
In CASIA-B dataset there are 3 size of testing. First we need to combine data from GaitGraph into 1 file [.npy] (they provide 2 csv files, train and test)
```
cd tools
python combine_data.py
```