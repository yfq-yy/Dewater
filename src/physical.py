import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import *
from torch.nn.modules.activation import Hardtanh
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from torch.nn import init
from networks import define_G, define_D
from torch.autograd import Variable
import functools

# -------------------------------------------------------------------------- #
#  Network Architecture
# -------------------------------------------------------------------------- #
class DecompModel(nn.Module):
    def __init__(self):
        super(DecompModel, self).__init__()
        self.fognet = FogNet(33)
        self.radiux = [2, 4, 8, 16, 32]
        self.eps_list = [0.001, 0.0001]
        self.eps = 0.001
        self.gf = None
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        lf = self.decomposition(x)
        trans, atm = self.fognet(torch.cat([x, lf], dim=1))
        return trans, atm  # trans, atm, clean

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
        #HF = torch.cat(HF_list, dim=1)
        return LF


class FogNet(nn.Module):
    def __init__(self, input_nc):
        super(FogNet, self).__init__()
        self.atmconv1x1 = nn.Conv2d(input_nc, input_nc, kernel_size=1, stride=1, padding=0)
        self.atmnet = define_D(input_nc, 64, 'n_estimator', n_layers_D=5, norm='batch', use_sigmoid=True, gpu_ids=[])
        self.transnet = TransUNet(input_nc, 1)
        self.htanh = nn.Hardtanh(0, 1)
        self.relu1 = ReLU1()
        self.relu = nn.ReLU()

    def forward(self, x, dx=0, dy=0):
        _, c, h, w = x.size()

        A = self.relu(self.atmnet(self.atmconv1x1(x[:, :, dy:dy+256, dx:dx+256])))
        trans = self.relu(self.transnet(x))
        atm = A.repeat(1, 1, h, w)
        trans = trans.repeat(1, 3, 1, 1)
        return trans, atm


class TransUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(TransUNet, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.inc = inconv(in_channel, 64)
        self.image_size = 256
        self.down1 = down(64, 128)  # 112x112 | 256x256 | 512x512
        self.down2 = down(128, 256)  # 56x56   | 128x128 | 256x256
        self.down3 = down(256, 512)  # 28x28   | 64x64  | 128x128
        self.down4 = down(512, 512)  # 14x14   | 32x32  | 64x64

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
# ---------------------------------------------------------------------------- #
# Sub-Modules
# ---------------------------------------------------------------------------- #


class ReLU1(Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU1, self).__init__(0, 1, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str



def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, dropout_rate=0.5):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x

        if self.pooling:
            x = self.pool(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', dropout_rate=0.5):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.dropout_rate = dropout_rate

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        # layer 1
        x = F.relu(self.conv1(x))
        # layer 2
        x = F.relu(self.conv2(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        return x

# ============================================================================
# sub-parts of the U-Net model

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            ## nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            ## nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


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
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'n_estimator':
        netD = NLayerEstimator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)




class NLayerEstimator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerEstimator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        sequence += [nn.AdaptiveAvgPool2d(1),
                     nn.Conv2d(ndf * nf_mult, 1024, kernel_size=1),
                     nn.LeakyReLU(0.2, True),
                     nn.Conv2d(1024, 3, kernel_size=1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

