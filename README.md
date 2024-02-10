# A Physics-based Attention CycleGAN for UAV Coastal Zone Unpaired Images Water Removal


### Update
2024.02.10: Upload the code.

### Abstract
Coastal zone ecosystems play a vital role in maintaining marine environmental stability, and Unmanned Aerial Vehicles (UAVs) are widely utilized for monitoring these ecosystems, analyzing objects within UAV-captured images is a crucial technology for marine life identification. However, UAV images are subject to image colour distortion and blurring caused by the interplay of ambient light and water. Therefore further image processing, e.g., water removal, is required to clearly identify objects. Currently, existing algorithms primarily target underwater image restoration or rain removal, often relying on paired datasets, which are challenging to obtain in real-world scenarios. Consequently, this research addresses the following core issues. (1) Water removal  based on unpaired ocean low-tide and high-tide images: effectively resolving the challenge of images water removal without paired data. (2) Preserving color and texture in the water removal process: ensuring that the water removal process maintains realism of the physical imaging process of the image. (3) Image water removal evaluation metrics: determining suitable evaluation criteria for assessing the quality of water-free images. To tackle these challenges, this paper introduces a physics-based attention CycleGAN for UAV coastal zone unpaired images water removal. This approach holds significant promise for the accurate identification of coastal zone organisms. Specifically, our approach involves two key steps: the unsupervised training of networks with unpaired data is accomplished through CycleGAN's recurrent network, which enables seamless domain transitions from low-tide level to high-tide level. Employing a physics-based attention guidance module to guide image restoration and maintain realism. Additionally, we utilize established image restoration evaluation metrics for qualitative analysis and conduct an extensive set of experiments on the water removal dataset to validate the method's effectiveness.

![image](https://github.com/OaDsis/DerainCycleGAN/blob/main/figures/model.png)
![image](https://github.com/OaDsis/DerainCycleGAN/blob/main/figures/result.png)

### Requirements
- python 3.6.10
- torch 1.4.0
- torchvision 0.5.0
- NVIDIA GeForce GTX GPU with 12GB memory at least, or you can change image size in option.py

### Datasets
- Rain100L
- Rain800
- Rain12
- SPA-Data
- Real-Data

You can download above datasets from [here](https://github.com/hongwang01/Video-and-Single-Image-Deraining#datasets-and-discriptions)

### Pre-trained Models
You can download pre-trained models from [here](https://drive.google.com/drive/folders/1DvOFGIdXXnNm1iage69HuasUPHZSXYYt?usp=sharing) and put them into corresponding folders, then the content is just like:

"./results/Rain100L/net_best_Rain100L.pth"

"./results/Rain800/net_best_Rain800.pth"

"./vgg16/vgg16.weight"

Note: **net_best_Rain100L.pth** is for the testing of Rain100L, Rain12, SPA-Data, and Real-Data datasets. **net_best_Rain800.pth** is for the testing of Rain800 dataset. **vgg16.weight** is for the parameters of Vgg16.

### Usage
#### Prepare dataset:
Taking training Rain100L as example. Download Rain100L (including training set and testing set) and put them into the folder "./datasets", then the content is just like:

"./datasets/rainy_Rain100L/trainA/rain-***.png"

"./datasets/rainy_Rain100L/trainB/norain-***.png"

"./datasets/test_rain100L/trainA/rain-***.png"

"./datasets/test_rain100L/trainB/norain-***.png"
#### Train (Take Rain100L dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/rainy_Rain100L --val_path ../datasets/test_rain100L --name Rain100L
```
#### Test (Take Rain100L dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py --test_path ../datasets --name Rain100L --resume ../results/Rain100L/net_best_Rain100L.pth --mode 1
```
you can change the mode to test different datasets, i.e., Rain100L = 1, Rain12 = 2, Real-Data = 3, Rain800 = 4, SPA-Data = 5.
#### Generate Rain Images
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 auto.py --auto_path ../datasets --name Auto100L --resume ../results/Rain100L/net_best_Rain100L.pth --mode 0 --a2b 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 auto.py --auto_path ../datasets --name Auto800 --resume ../results/Rain800/net_best_Rain800.pth --mode 1 --a2b 0
```
### Citation
Please cite our paper if you find the code useful for your research.
```
@article{wei2021deraincyclegan,
  title={Deraincyclegan: Rain attentive cyclegan for single image deraining and rainmaking},
  author={Wei, Yanyan and Zhang, Zhao and Wang, Yang and Xu, Mingliang and Yang, Yi and Yan, Shuicheng and Wang, Meng},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={4788--4801},
  year={2021},
  publisher={IEEE}
}
```
### Acknowledgement
Code borrows from [DRIT](https://github.com/HsinYingLee/DRIT) by Hsin-Ying Lee. Thanks for sharing !

### Contact
Thanks for your attention. If you have any questions, please contact my email: weiyy@hfut.edu.cn. 
