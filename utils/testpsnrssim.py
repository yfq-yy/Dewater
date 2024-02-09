
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
 
from ssim import to_ssim_skimage, to_psnr
 
root1 = "/data/fyao309/dewater/outputs/lsui_cyclegan_b8"
root2 = "/data/fyao309/dewater/datasets/underwater_test"
psnr_list = []
ssim_list = []
class datasets(Dataset):
 
    def __init__(self):
        self.X = os.listdir(os.path.join(root1, "pre49"))
        self.X.sort()
        self.Y = os.listdir(os.path.join(root2, "trainB1"))
        self.Y.sort()
 
    def __getitem__(self, index):
        a = self.X[index]
        b = self.Y[index]
        print(a,b)
        x_path = os.path.join(root1, "pre49", a)
        y_path = os.path.join(root2, "trainB1", b)
        pathX = Image.open(x_path)
        pathY = Image.open(y_path)
        pathX_tensor= transforms.CenterCrop(256)(pathX)
        pathY_tensor = transforms.CenterCrop(256)(pathY)
        pathX_tensor = transforms.ToTensor()(pathX_tensor)
        pathY_tensor = transforms.ToTensor()(pathY_tensor)
        return pathX_tensor, pathY_tensor
 
    def __len__(self):
        return len(os.listdir(os.path.join(root1, "pre49")))
 
 
dataset = datasets()
 
 
my_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
 
for x, y in my_dataloader:
    psnr_list.extend(to_psnr(x, y))
    ssim_list.extend(to_ssim_skimage(x, y))
    print(to_psnr(x, y))
avr_psnr = sum(psnr_list) / len(psnr_list)
avr_ssim = sum(ssim_list) / len(ssim_list)
print(avr_psnr, avr_ssim)

