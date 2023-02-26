# DenseUnet
Modified U-Net for pavement crack segmentation. This repository adopted codes from https://github.com/VainF/DeepLabV3Plus-Pytorch.
  
## Quick Start  
### 1. Available architechtures  

| Name    |  Description        |
| :---: | :---:     |
|UNet|-|
|DenseUnetv1|Extra skip connections in both encoder and decoder|
|DenseUnetv2|Extra skip connections in encoder only ||  
  
### 2. Train a model from stratch  
To train a model using DenseUnetv1 as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_denseunetv1' --dataset 'crack' total_epochs 100
```
To train a model using DenseUnetv2 as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_denseunetv2' --dataset 'crack' total_epochs 100
```
