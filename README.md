# DewaterGAN: An Physics-Guided Unsupervised Image Water Removal for UAV in Coastal Zone


### Update
2024.02.10: Upload the code.

### Abstract
Accurate recognition of marine species in drone-captured images is essential in maintaining the stability of coastal zone ecosystems. Unmanned Aerial Vehicle (UAV) remote sensing images usually lack paired supervised signals and suffer from color distortion and blurring due to interaction of ambient light with cross-medium transmission between air and water. However, current algorithms mainly focus on supervised training methods and also ignore the interaction involved in the cross-medium transmission of light in water. In this paper, for UAV in coastal zones, we propose an unsupervised image water removal model--DewaterGAN, which is based solely on low-tide and high-tide images without paired supervised signals and also preserves color and texture in the water removal process. Specifically, our approach involves two key steps: an unsupervised training CycleGAN network accomplishes domain transitions from low-tide level to high-tide level, and a physics-based attention module guides image water removal and maintains authenticity. Additionally, we utilize evaluation metrics of image restoration  PSNR and SSIM to qualitatively analyze the performance of the model. We also employed several non-reference metrics (UIQM, UCIQE, NIQE, BRISQUE, LIQE, ILNIQE and CLIPIQA) to evaluate the visual quality of the image de-watering process. Extensive experiments conducted on both our water removal dataset and public datasets validate the efficacy of our model. 

![image](https://github.com/yfq-yy/Dewater/blob/master/figures/pipline.png)


### Requirements
- python 3.8.18
- torch 1.13.0
- torchvision 0.14.0
- scikit-image 0.18.3
  
### Datasets
- UAV-Water
- LSUI
- UIEB
- Rain100L
- SPA-Data

You can download above datasets from [here](XXX)

### Pre-trained Models
You can download pre-trained models from [here](XXX) and put them into corresponding folders, then the content is just like:

"./results/UAVWater/net_best_00299.pth"

"./results/Rain100L/net_best_00300.pth"

"./vgg16/vgg16.weight"

Note: **vgg16.weight** is for the parameters of Vgg16.

### Usage
#### Prepare dataset:
Taking training Rain100L as example. Download UAV-Water and Rain100L (including training set and testing set) and put them into the folder "./datasets", then the content is just like:

"./datasets/UAVwater_train/trainA/rain-***.png"

"./datasets/UAVwater_test/trainB/norain-***.png"

"./datasets/rain100L_train/trainA/rain-***.png"

"./datasets/rain100L_test/trainB/norain-***.png"
#### Train (Take UAV-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/UAVwater_train --val_path ../datasets/UAVwater_test --name UAVWater
```
#### Resume Train (Take UAV-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/UAVwater_train --val_path ../datasets/UAVwater_test --name UAVWater --resume ../results/UAVWater/net_best_00299.pth
```
#### Test (Take UAV-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --test_path ../datasets --name UAVWater --resume ../results/UAVWater/net_best_00299.pth --mode 1
```
#### val (Take UAV-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0 python3 val.py --val_path ../datasets/UAVwater_test --name UAVWater --resume ../results/UAVWater/net_best_00299.pth
```
#### valpair (Take Rain100L dataset as example):
```
cd /utils
change path of derain/reference images
python testpsnrssim.py
```
![image](https://github.com/yfq-yy/Dewater/blob/master/figures/waterimage1.png)
### Contact
Thanks for your attention. If you have any questions, please contact my email: yaofengqin8312@stu.ouc.edu.cn. 
