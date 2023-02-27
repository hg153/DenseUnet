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

### 2. Dataset  
Ground-truth image has only two values, with 0 indicating background and 1 indicating crack. Datasets should be stored in `datasets\data\your dataset name` folder and follow the structure as shown below.  
![image](https://user-images.githubusercontent.com/58408775/221439479-3c694de9-1de8-4b57-b541-e8a504a666a5.png)

### 3. Train a model from stratch  
To train a model using UNet as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_unet' --dataset 'crack' --total_epochs 100
```
To train a model using DenseUnetv1 as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_denseunetv1' --dataset 'crack' --total_epochs 100
```
To train a model using DenseUnetv2 as backbone:  
```bash 
python main.py --data_root './datasets/data/sample_dataset' --model 'fcn_denseunetv2' --dataset 'crack' --total_epochs 100
```
Trained models will be saved in the `checkpoints` folder.  
  
### 4. Evaluation  
Use the following command to evaluate the trained DenseUnetv1 model. Due to the imprecise crack annotation, a strategy of relaxation can be implemented, where predicted crack pixels within 2 pixels from the ground-truth will be considered correct detection.   
```bash
python eval.py --input './datasets/data/sample_dataset' --model 'fcn_denseunetv1' --relaxation True --ckpt 'your trianed model'
```  
  
### 5. Prediction  
Use the following command to predict images in a folder.  
```bash
python predict.py --input 'datasets/data/sample_dataset/images/val' --dataset 'crack' --model 'fcn_denseunetv1' --ckpt 'denseunetv1.pth' --save_val_results_to 'results'
``` 
