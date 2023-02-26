# DenseUnet
Modified U-Net for pavement crack segmentation. This repository adopted codes from https://github.com/VainF/DeepLabV3Plus-Pytorch.
  
## Quick Start  
### 1. Available architechtures  

| Name    |  Description        |
| :---: | :---:     |
|UNet|-|
|DenseUnetv1|Extra skip connections in both encoder and decoder|
|DenseUnetv2|Extra skip connections in encoder only ||  

Note: The UNet is modified with the implementation of paddings. Feature maps do no shrink due to convolution operations.  

### Dataset  
Ground-truth image has only two values, with 0 indicating background and 1 indicating crack. Datasets should be stored in `datasets\data` folder and follow the structure as shown below.  
![image](https://user-images.githubusercontent.com/58408775/221439444-07cd0a5a-9ae3-493b-b07c-d25e34ca3f1f.png)

### 3. Train a model from stratch  
To train a model using UNet as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_unet' --dataset 'crack' total_epochs 100
```
To train a model using DenseUnetv1 as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_denseunetv1' --dataset 'crack' total_epochs 100
```
To train a model using DenseUnetv2 as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_denseunetv2' --dataset 'crack' total_epochs 100
```
