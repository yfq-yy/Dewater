# DewaterGAN: An Unsupervised Image Water Removal for UAV in Coastal Zone


### Update
2024.02.10: Upload the code.

### Abstract
Coastal zone ecosystems play a vital role in maintaining marine environmental stability, and Unmanned Aerial Vehicles (UAVs) are widely utilized for monitoring these ecosystems, analyzing objects within UAV-captured images is a crucial technology for marine life identification. However, UAV images are subject to image colour distortion and blurring caused by the interplay of ambient light and water. Therefore further image processing, e.g., water removal, is required to clearly identify objects. Currently, existing algorithms primarily target underwater image restoration or rain removal, often relying on paired datasets, which are challenging to obtain in real-world scenarios. Consequently, this research addresses the following core issues. (1) Water removal  based on unpaired ocean low-tide and high-tide images: effectively resolving the challenge of images water removal without paired data. (2) Preserving color and texture in the water removal process: ensuring that the water removal process maintains realism of the physical imaging process of the image. (3) Image water removal evaluation metrics: determining suitable evaluation criteria for assessing the quality of water-free images. To tackle these challenges, this paper introduces a physics-based attention CycleGAN for UAV coastal zone unpaired images water removal. This approach holds significant promise for the accurate identification of coastal zone organisms. Specifically, our approach involves two key steps: the unsupervised training of networks with unpaired data is accomplished through CycleGAN's recurrent network, which enables seamless domain transitions from low-tide level to high-tide level. Employing a physics-based attention guidance module to guide image restoration and maintain realism. Additionally, we utilize established image restoration evaluation metrics for qualitative analysis and conduct an extensive set of experiments on the water removal dataset to validate the method's effectiveness.

![image](https://github.com/yfq-yy/Dewater/blob/master/figures/model.png)


### Requirements
- python 3.8.18
- torch 1.13.0
- torchvision 0.14.0
- scikit-image 0.18.3
  
### Datasets
- OUC-Water
- LSUI
- UIEB
- Rain100L
- SPA-Data

You can download above datasets from [here](XXX)

### Pre-trained Models
You can download pre-trained models from [here](XXX) and put them into corresponding folders, then the content is just like:

"./results/OUCWater/net_best_00299.pth"

"./results/Rain100L/net_best_00300.pth"

"./vgg16/vgg16.weight"

Note: **vgg16.weight** is for the parameters of Vgg16.

### Usage
#### Prepare dataset:
Taking training Rain100L as example. Download OUC-Water and Rain100L (including training set and testing set) and put them into the folder "./datasets", then the content is just like:

"./datasets/OUCwater_train/trainA/rain-***.png"

"./datasets/OUCwater_test/trainB/norain-***.png"

"./datasets/rain100L_train/trainA/rain-***.png"

"./datasets/rain100L_test/trainB/norain-***.png"
#### Train (Take OUC-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/OUCwater_train --val_path ../datasets/OUCwater_test --name OUCWater
```
#### Resume Train (Take OUC-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/OUCwater_train --val_path ../datasets/OUCwater_test --name OUCWater --resume ../results/OUCWater/net_best_00299.pth
```
#### Test (Take OUC-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --test_path ../datasets --name OUCWater --resume ../results/OUCWater/net_best_00299.pth --mode 1
```
#### val (Take OUC-Water dataset as example):
```
CUDA_VISIBLE_DEVICES=0 python3 val.py --val_path ../datasets/OUCwater_test --name OUCWater --resume ../results/OUCWater/net_best_00299.pth
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
