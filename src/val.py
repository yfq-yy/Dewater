import torch
from options import TestOptions
from dataset import dataset_unpair_val
from model import DerainCycleGAN
from saver import save_imgs
import os
from SSIM import *
from utils import *

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()
  criterion = SSIM()
  criterion.cuda(opts.gpu)
  # data loader
  dataset_val = dataset_unpair_val(opts)
  loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=opts.nThreads)    
  print("********************************")
  print(loader_val)

  # model
  print('\n--- load model ---')
  model = DerainCycleGAN(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()
  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)
  valLogger = open('%s/valpsnr&ssim.log' % os.path.join(opts.result_dir, opts.name), 'w')



  # val
  print('\n--- valing ---')
  # for idx1, (img1, needcrop,imgname) in enumerate(loader):
  #   print('{}/{}'.format(idx1, len(loader)))
  #   img1 = img1.cuda()
  #   imgs = []
  #   imgname = str(imgname).split("/")[4].split(".")[0] #yfq
  #   names = []
  #   for idx2 in range(1):
  #     with torch.no_grad():
  #       img = model.test_forward(img1, a2b=opts.a2b)
  #     imgs.append(img)
  #     names.append('{}'.format(imgname))

  #   #print("-------************---")
  #   save_imgs(imgs, names, os.path.join(result_dir), needcrop)
  ssim_sum=0
  ssim_avg=0
  psnr_sum=0
  psnr_avg=0
  for i, (input_val, target_val) in enumerate(loader_val, 0):        
        
      input_val, target_val = input_val.cuda(opts.gpu), target_val.cuda(opts.gpu)  
      #imaA->genA->genB
      out_val_A = model.test_forward(input_val, a2b=opts.a2b)
      out_val_A = model.test_forward_revert(out_val_A,a2b=opts.a2b)
      #imaB->genB->genA
      out_val_B = model.test_forward_revert(target_val, a2b=opts.a2b)
      out_val_B = model.test_forward(out_val_B,a2b=opts.a2b)
        
      #ssim_val = criterion(target_val, out_val)
      ssim_val_A = criterion(input_val, out_val_A)
      ssim_val_B = criterion(target_val, out_val_B)
      ssim_val =  (ssim_val_A + ssim_val_B)/2
      ssim_sum = ssim_sum + ssim_val.item()
      #yfq
      # input_val = torch.clamp(input_val, 0., 1.)
      # target_val = torch.clamp(target_val, 0., 1.)
      input_val = torch.clamp(input_val, 0., 1.)
      target_val = torch.clamp(target_val, 0., 1.)
      out_val_A = torch.clamp(out_val_A, 0., 1.)
      out_val_B = torch.clamp(out_val_B, 0., 1.)
      psnr_val_A = batch_PSNR(out_val_A, input_val, 1.) 
      psnr_val_B = batch_PSNR(out_val_B, target_val, 1.) 
      psnr_val = (psnr_val_A + psnr_val_B)/2
      psnr_sum = psnr_sum + psnr_val

      print("[%d/%d] psnr: %.4f, ssim: %.4f" %
              (i+1, len(loader_val), psnr_val, ssim_val.item()))  
      
      valLogger.write('%03d\t%04f\t%04f\r\n' % \
                  (i+1, psnr_val, ssim_val))
      valLogger.flush()

                          
  ssim_avg = ssim_sum/len(loader_val)
  psnr_avg = psnr_sum/len(loader_val)
  print("psnr_avg: %.4f, ssim_avg: %.4f" % (psnr_avg, ssim_avg))  
  valLogger.write('totalavg:%03d\t%04f\t%04f\r\n' % \
                  (i+1, psnr_avg, ssim_avg))
  valLogger.flush()
  valLogger.close()

  return

if __name__ == '__main__':
  main()
