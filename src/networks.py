import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os
import torchvision.models as models
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from utils import *
from torch.nn import init
####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class MultiScaleDis(nn.Module):
  def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
    super(MultiScaleDis, self).__init__()
    ch = 64
    self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    self.Diss = nn.ModuleList()
    for _ in range(n_scale):
      self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
    tch = ch
    for _ in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
      tch *= 2
    if sn:
      model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
    else:
      model += [nn.Conv2d(tch, 1, 1, 1, 0)]
    return nn.Sequential(*model)

  def forward(self, x):
    outs = []
    for Dis in self.Diss:
      outs.append(Dis(x))
      x = self.downsample(x)
    return outs


####################################################################
#--------------------------- double_conv ----------------------------
####################################################################
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            ## nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            ## nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

####################################################################
#--------------------------- inconv ----------------------------
####################################################################
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
    
####################################################################
#--------------------------- up ----------------------------
####################################################################
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffY // 2, int(diffY / 2), diffX // 2, int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
####################################################################
#--------------------------- down ----------------------------
####################################################################
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

####################################################################
#--------------------------- outconv ----------------------------
####################################################################
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,1)

    def forward(self, x):
        x = self.conv(x)
        return x
####################################################################
#--------------------------- TransUNet ----------------------------
####################################################################    
class TransUNet(nn.Module):
    def __init__(self):
        super(TransUNet, self).__init__()
        self.conv1x1 = nn.Conv2d(33, 32, kernel_size=1, stride=1, padding=0)
        self.inc = inconv(32, 64)
        #self.image_size = 256
        self.down1 = down(64, 128)  # 112x112 | 256x256 | 512x512
        self.down2 = down(128, 256)  # 56x56   | 128x128 | 256x256
        self.down3 = down(256, 512)  # 28x28   | 64x64  | 128x128
        self.down4 = down(512, 512)  # 14x14   | 32x32  | 64x64
        n_classes = 1
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.relu(self.outc(x))
        return x



####################################################################
#--------------------------- Atmnet ----------------------------
#self.atmnet = define_D(input_nc, 64, 'n_estimator', n_layers_D=5, norm='batch', use_sigmoid=True, gpu_ids=[])
####################################################################   
class Atmnet(nn.Module):
    def __init__(self):
        super(Atmnet, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        use_bias = norm_layer.func == nn.InstanceNorm2d
        input_nc =33
        kw = 4
        padw = 1
        ndf=64
        self.use_sigmoid=False

        self.atmconv1x1 = nn.Conv2d(input_nc, input_nc, kernel_size=1, stride=1, padding=0)
        self.atm_conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
            )
        self.atm_conv2 = nn.Sequential(
            nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
            )
      
        self.atm_conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
            )
        self.atm_conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
            )
        self.atm_conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
            )
        self.atm_conv6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=1)
            )
    
        self.atm_sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #batch_size, row, col = input.size(0), input.size(2), input.size(3)
       
        x = self.atmconv1x1(x)
        x_1 = self.atm_conv1(x)
        x_2 = self.atm_conv2(x_1)
        x_3 = self.atm_conv3(x_2)
        x_4 = self.atm_conv4(x_3)
        x_5 = self.atm_conv5(x_4)
        x_6 = self.atm_conv6(x_5)
        if self.use_sigmoid:
           x= self.atm_sig(x_6)
        else:
           x = self.relu(x_6)
        
        return x

####################################################################
#--------------------------- STREAK ---------------------------
####################################################################
class RainNet(nn.Module):
    def __init__(self):
        super(RainNet, self).__init__()
        in_ch = 33
        out_ch = 1
        self.conv1x1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.block1 = ResidualBlockUp(in_ch, 256)
        self.block2 = ResidualBlockStraight(256, 256)
        self.block3 = ResidualBlockStraight(256, 256)
        self.block4 = ResidualBlockDown(256, out_ch, last=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

class ResidualBlockStraight(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, last=False):
        super(ResidualBlockStraight, self).__init__()
        assert (in_channels == out_channels)
        self.conv1 = res_conv(in_channels, 64, dil=dilation)
        self.conv2 = res_conv(64, 64)
        self.conv3 = res_conv(64, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, last=True):
        super(ResidualBlockDown, self).__init__()
        self.conv1 = res_conv(in_channels, in_channels, dil=dilation)
        self.conv2 = res_conv(in_channels, 128)
        self.conv3 = res_conv(128, 64)
        self.conv_out = nn.Conv2d(320, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = torch.cat((out, residual), dim=1)
        out = self.relu(out)
        out = self.conv_out(out)
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super(ResidualBlockUp, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv1 = res_conv(64, 64)
        self.conv2 = res_conv(64, 64)
        self.conv3 = res_conv(64, 256)
        self.conv_in = nn.Conv2d(64, 256, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        x = self.relu(self.conv0(x))
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        residual = self.conv_in(residual)
        out = out + residual
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


def res_conv(in_ch, out_ch, k=3, s=1, dil=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, dilation=dil, padding=1, bias=bias)

####################################################################
#--------------------------- URAD ----------------------------
####################################################################
class PBA(nn.Module):
    def __init__(self):
        super(PBA, self).__init__()
        self.radiux = [2, 4, 8, 16, 32]
        self.eps_list = [0.001, 0.0001]
        self.gf = None
        self.atmnet = Atmnet()
        self.transnet1 = TransUNet()
        self.transnet2 = TransUNet()
        self.rainnet = RainNet()
        
       
    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        #***********yfq********************
        lf, hf = self.decomposition(input)
        A = self.atmnet(torch.cat([input, lf], dim=1))
        atm = A.repeat(1, 1, row, col)
        trans1 = self.transnet1(torch.cat([input, lf], dim=1))
        trans2 = self.transnet2(torch.cat([input, lf], dim=1))
        streak = self.rainnet(torch.cat([input, hf], dim=1))
        #streak = streak.repeat(1, 3, 1, 1)
        #input = torch.cat((input, mask), 1) 
        #input = torch.cat((input, hf), 1)
        #yfq
        size = mask.size()[2:][0],mask.size()[2:][1]
        trans1 = F.interpolate(trans1,size)
        trans2 = F.interpolate(trans2,size)
        mask = trans1 * trans2 * streak + (1-trans1*trans2) * atm
    
        return mask
   
    def decomposition(self, x):
        LF_list = []
        HF_list = []
        res = get_residue(x)
        res = res.repeat(1, 3, 1, 1)
        for radius in self.radiux:
            for eps in self.eps_list:
                self.gf = GuidedFilter(radius, eps)
                LF = self.gf(res, x)
                LF_list.append(LF)
                HF_list.append(x - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF

####################################################################
#--------------------------- Generators ----------------------------
####################################################################
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation = 2),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation = 4),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation = 8),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation = 16),
            nn.ReLU()
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
            )

    def forward(self, input, mask):
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
    
        res1 = x
        x = self.conv2(x)
        # pdb.set_trace()            
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
  
        
        diffX = x.size()[2] - res2.size()[2]
        diffY = x.size()[3] - res2.size()[3]
        res2 = F.pad(res2, (diffY // 2, diffY- diffY // 2, diffX // 2, diffX- diffX // 2))
      
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)

        diffX = x.size()[2] - res1.size()[2]
        diffY = x.size()[3] - res1.size()[3]
        res1 = F.pad(res1, (diffY // 2, diffY- diffY // 2, diffX // 2, diffX- diffX // 2))
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return frame1, frame2, x

####################################################################
#--------------------------- Vgg16 ----------------------------
####################################################################
class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        relu4_4 = h           
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        return relu3_3

def init_vgg16(model_folder):
	"""load the vgg16 model feature"""
	if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
		if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
			os.system(				# 下载vgg16.t7文件
				'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
		vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
		vgg = Vgg16()
		for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
			dst.data[:] = src
		torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))

def define_vgg(path):
    vgg = Vgg16()
    init_vgg16(path)
    vgg.load_state_dict(torch.load(os.path.join(path, "vgg16.weight")))
    vgg = vgg.cuda()
    return vgg

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += conv3x3(inplanes, outplanes)
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer

def conv3x3(in_planes, out_planes):
  return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

####################################################################
#--------------------- Spectral Normalization ---------------------
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))