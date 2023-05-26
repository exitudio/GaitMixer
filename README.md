

# GaitMixer
This repository contains the PyTorch code for:

__GaitMixer: Skeleton-based Gait Representation Learning via Wide-spectrum Multi-axial Mixer__ 
The paper is accepted to [ICASSP 2023](https://2023.ieeeicassp.org/).

[Ekkasit Pinyoanuntapong](https://github.com/exitudio), Ayman Ali, Pu Wang, Minwoo Lee, Chen Chen


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gaitmixer-skeleton-based-gait-representation/multiview-gait-recognition-on-casia-b)](https://paperswithcode.com/sota/multiview-gait-recognition-on-casia-b?p=gaitmixer-skeleton-based-gait-representation)
[![arxiv](https://img.shields.io/badge/arXiv:2210.15491-red)](https://arxiv.org/abs/2210.15491) 


<!-- [![DOI:10.1109/ICIP42928.2021.9506717](https://img.shields.io/badge/DOI-10.1109%2FICIP42928.2021.9506717-blue)](https://doi.org/10.1109/ICIP42928.2021.9506717) [![BibTeX](https://img.shields.io/badge/cite-BibTeX-yellow)](#CitingGaitGraph) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gaitgraph-graph-convolutional-network-for/multiview-gait-recognition-on-casia-b)](https://paperswithcode.com/sota/multiview-gait-recognition-on-casia-b?p=gaitgraph-graph-convolutional-network-for) -->



![Pipeline](assets/GaitMixer-diagram.jpg)
## Campare to Previous SOTA Skeleton-based Gait Recognition
We proposed 2 SOTA methods GaitFormer and GaitMixer in skeleton based gait recognition, improving from GaitGraph by 12% on average.
|      Method      |  NM  |  BG  |  CL  |  Mean  |
|-----------------:|-----:|-----:|-----:|-------:|
|PoseGait          | 68.7 | 44.5 | 36.0 |  49.7  |
|GaitGraph         | 87.7 | 74.8 | 66.3 |  76.3  |
|GaitGraph2        | 82.0 | 73.2 | 63.6 |  72.9  |
|<b>GaitFormer (ours)</b> | 91.5 | 81.4 | 77.2 |  83.4  |
|<b>GaitMixer (ours)</b> | <b>94.9</b> | <b>85.6</b> | <b>84.5</b> |  <b>88.3</b>  |

## Quick Start

First, create a virtual environment or install dependencies directly with:
```shell
conda env create -f environment.yml
```

## Data preparation
Follow [GaitGraph data preparation](https://github.com/tteepe/GaitGraph#data-preparation)

## Train
To train the model you can run the `train.py` script. Our paper presents 2 models to study behavior of self-attetion and large kernel depthwise separable convolution
1. ***GaitMixer*** (Spatial self-attention & temporal convolution)
```shell
python train.py casia-b ../data/casia-b_pose_train_valid.csv --valid_data_path ../data/casia-b_pose_test.csv
```
The pre-trained models is available at [GaitMixer](https://github.com/exitudio/GaitMixer/releases/download/pretrain/GaitMixer.pt)

2. ***GaitFormer*** (Spatial-temporal self-attention)
```shell
python train.py casia-b ../data/casia-b_pose_train_valid.csv --valid_data_path ../data/casia-b_pose_test.csv --model_type spatiotemporal_transformer
```
The pre-trained models is available at [GaitFormer](https://github.com/exitudio/GaitMixer/releases/download/pretrain/GaitFormer.pt).

Check `experiments/*.sh` to see the example of other configurations used in the paper. 
By default, testing runs every 10 epochs. Use ```--test_epoch_interval 1``` to change number of epochs per testing .
See more training options in [training doc](./docs/train.md)



## Main Results
Top-1 Accuracy per probe angle excluding identical-view cases for the provided models on 
[CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) dataset.

|        |    0 |   18 |   36 |   54 |   72 |   90 |   108 |   126 |   144 |   162 |   180 |   mean |
|:-------|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|------:|-------:|
| NM#5-6 | 94.4 | 94.9 | 94.6 | 96.3 | 95.3 | 96.3 | 95.3  | 94.7  |  95.3 |  94.7 |  92.2 |   94.9 |
| BG#1-2 | 83.5 | 85.6 | 88.1 | 89.7 | 85.2 | 87.4 | 84.0  | 84.7  |  84.6 |  87.0 |  81.4 |   85.6 |
| CL#1-2 | 81.2 | 83.6 | 82.3 | 83.5 | 84.5 | 84.8 | 86.9  | 88.9  |  87.0 |  85.7 |  81.6 |   84.5 |


## Licence & Acknowledgement
GaitMixer itself is released under the MIT License (see LICENSE).

The following parts of the code are borrowed from other projects. Thanks for their wonderful work!
- GaitGraph: [tteepe/GaitGraph](https://github.com/tteepe/GaitGraph)
- PoseFormer: [zczcwh/PoseFormer](https://github.com/zczcwh/PoseFormer)

---
If you have any question feel free to contact me at exitudio@gmail.com or epinyoan@uncc.edu

https://ekkasit.com

## <a name="CitingGaitMixer"></a>Citing GaitMixer
If you find our work useful in your research, please consider citing:

```
@INPROCEEDINGS{10096917,
  author={Pinyoanuntapong, Ekkasit and Ali, Ayman and Wang, Pu and Lee, Minwoo and Chen, Chen},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Gaitmixer: Skeleton-Based Gait Representation Learning Via Wide-Spectrum Multi-Axial Mixer}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096917}}
```
