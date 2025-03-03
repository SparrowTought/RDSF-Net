import torch
import torch.nn as nn
from torch import nn, einsum
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from .CMamba import CMamba

from einops import rearrange

from models.BiTFP import BiTFP, DFF
import numpy as np
import matplotlib.pyplot as plt
import math

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'RDSFNet':
        net = RDSFNet(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned')


    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


class up(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(up, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2,stride=2)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class down(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_planes, out_planes,3,1,1)
        self.bn = nn.BatchNorm2d(out_planes)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.bn(x)

        return x

class Conv_yuan(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv_yuan, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim,3,1,1)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class RDSFNet(nn.Module):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """

    def __init__(self, input_nc, output_nc, with_pos, normal_init=True,
                 ):
        super(RDSFNet, self).__init__()
        # self.resnet = resnet34()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-333f7ec4.pth'))
        self.backbone = CMamba()  # [64, 128, 320, 512]
        self.sigmoid = nn.Sigmoid()

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsamplex16 = nn.Upsample(scale_factor=16, mode='bilinear')

        # 64, 96, 192, 384

        # 64, 96, 192, 384
        self.BiTFP1 = BiTFP(384)
        self.BiTFP2 = BiTFP(192)
        self.BiTFP3 = BiTFP(96)
        self.BiTFP4 = BiTFP(64)

        self.decode4 = DFF(192)
        self.decode3 = DFF(96)
        self.decode2 = DFF(64)

        self.Up256 = up(384, 192)
        self.Up128 = up(192, 96)
        self.Up64 = up(96, 64)


        self.final = nn.Sequential(
            Conv_yuan(64, 32, 3, bn=True, relu=True),
            Conv_yuan(32, 2, 3, bn=False, relu=False)
        )
        self.final2 = nn.Sequential(
            Conv_yuan(96, 32, 3, bn=True, relu=True),
            Conv_yuan(32, 2, 3, bn=False, relu=False)
        )
        self.final3 = nn.Sequential(
            Conv_yuan(192, 32, 3, bn=True, relu=True),
            Conv_yuan(32, 2, 3, bn=False, relu=False)
        )
        self.num_images = 0
        if normal_init:
            self.init_weights()


    def forward(self, x1, x2):
        outputs = []
        pvt, pvt2 = self.backbone(x1,x2)

        x_c1_1p, x_c1_2p, x_c1_3p, x_c1_4p = pvt[0], pvt[1], pvt[2], pvt[3]
        y_c1_1p, y_c1_2p, y_c1_3p, y_c1_4p = pvt2[0], pvt2[1], pvt2[2], pvt2[3]

        cross5 = self.BiTFP1(y_c1_4p, x_c1_4p)
        cross4 = self.BiTFP2(y_c1_3p, x_c1_3p)
        cross3 = self.BiTFP3(y_c1_2p, x_c1_2p)
        cross2 = self.BiTFP4(y_c1_1p, x_c1_1p)

        AFF4 = self.decode4(cross4, self.Up256(cross5))
        AFF3 = self.decode3(cross3, self.Up128(AFF4))
        AFF2 = self.decode2(cross2, self.Up64(AFF3))


        cross_out = self.upsamplex4(AFF2)
        cross_out2 = self.upsamplex8(AFF3)
        cross_out3 = self.upsamplex16(AFF4)


        out = self.final(cross_out)
        out2 = self.final2(cross_out2)
        out3 = self.final3(cross_out3)

        return out, out2, out3

    def init_weights(self):
        self.BiTFP1.apply(init_weights)
        self.BiTFP2.apply(init_weights)
        self.BiTFP3.apply(init_weights)
        self.BiTFP4.apply(init_weights)

        self.decode4.apply(init_weights)
        self.decode3.apply(init_weights)
        self.decode2.apply(init_weights)
        self.final.apply(init_weights)
        self.final3.apply(init_weights)
        self.final2.apply(init_weights)
