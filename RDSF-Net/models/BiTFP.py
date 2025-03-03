import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import pywt
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.max_pool(x) + self.avg_pool(x)
        out = self.fc2(self.relu1(self.fc1(out)))
        # 结果相加
        out = self.sigmoid(out)
        return out
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out =  a_w * a_h

        return out

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight1 = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias1 = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.weight2 = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        x1 = x1.view(B, self.group_num, -1)
        x2 = x2.view(B, self.group_num, -1)
        x = torch.cat((x1, x2), -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x1, x2 = x.chunk(2, dim=-1)
        x1 = x1.reshape(B, C, H, W)
        x2 = x2.reshape(B, C, H, W)
        return x1, x2


class BiTFP(nn.Module):
    def __init__(self, in_d):
        super(BiTFP, self).__init__()
        self.in_d = in_d
        self.out_d = in_d

        self.FHE = FHE(in_d)
        self.DCS = DiffCompletionSensor(in_d)

    def forward(self, x1, x2):
        eg1, eg2 = self.FHE(x1, x2)

        # difference enhance
        x = self.DCS(eg1, eg2)

        return x


class GaussianDenoiseFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(GaussianDenoiseFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)

    def create_gaussian_kernel(self, kernel_size, sigma):
        ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return kernel / torch.sum(kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')

        patches = x_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = patches.contiguous().view(B, C, H, W, -1)

        kernel = self.kernel.view(1, 1, 1, 1, -1).to(x.device)
        output = torch.sum(patches * kernel, dim=-1)

        return output



class FHE(nn.Module):
    def __init__(self, vit_dim):
        super(FHE, self).__init__()

        self.SelectiveConv = SelectiveConv(3, 1, False, vit_dim, vit_dim, first=False)


    def forward(self, x, y):
        output1, output2 = self.SelectiveConv(x, y)

        return output1, output2


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FrequencyDomain(nn.Module):
    def __init__(self, nc):
        super(FrequencyDomain, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class BidomainNonlinearMapping(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        self.spatial_process = nn.Sequential(nn.Conv2d(in_nc, in_nc, 1),
                                             nn.Sigmoid())
        self.FrequencyDomain = FrequencyDomain(in_nc)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        x_freq = self.FrequencyDomain(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_out = x * self.spatial_process(x_freq_spatial)

        return x_out

class MSFD(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim,dim,1)
        self.bid0 = FrequencyDomain(dim)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bid1 = FrequencyDomain(dim)
        self.conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.bid2 = FrequencyDomain(dim)
        self.conv2 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.convsimple = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                        nn.BatchNorm2d(dim),
                                        nn.ReLU(inplace=True))
        self.groups = dim

    def forward(self,  x):
        B, C, H, W = x.shape

        attn = self.conv(x)

        attn_0 = self.conv0(attn)
        attn_0 = self.bid0(attn_0)

        attn_1 = self.conv1(attn)
        attn_1 = self.bid1(attn_1)
        attn_2 = self.conv2(attn)
        attn_2 = self.bid2(attn_2)
        attn = attn_0 + attn_1 + attn_2
        out = self.convsimple(attn) + x

        return out
class SelectiveConv(nn.Module):
    def __init__(self, kernel_size, padding, bias, in_channels, out_channels, first=False):
        super(SelectiveConv, self).__init__()
        self.first = first
        self.conv1x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv3x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.conv1y = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2y = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv3y = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.INx = nn.InstanceNorm2d(in_channels)
        self.BNx = nn.BatchNorm2d(in_channels)

        self.INy = nn.InstanceNorm2d(in_channels)
        self.BNy = nn.BatchNorm2d(in_channels)

        self.GN = GroupBatchnorm2d(in_channels)

        self.relux = nn.LeakyReLU(inplace=True)
        self.reluy = nn.LeakyReLU(inplace=True)

        self.Dconx = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.Dcony = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))


    def forward(self, x, y):
        f_input = self.BNx(x)
        outx1 = self.conv1x(f_input)

        s_input = self.INx(x)
        outx2 = self.conv2x(s_input)
        resx = outx2 + outx1

        g_input = self.BNy(y)
        outy1 = self.conv1y(g_input)

        q_input = self.INy(y)
        outy2 = self.conv2y(q_input)
        resy = outy1 + outy2

        x_input, y_input = self.GN(resx, resy)
        outx = self.conv3x(x_input)
        outy = self.conv3y(y_input)

        outx = self.Dconx(outx) + x
        outy = self.Dcony(outy) + y

        return outx, outy


class Mask(nn.Module):


    def __init__(self, dim):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, 2, 1),
        )
        self.out_CA = CoordAtt(dim)
        self.out_mask = nn.Sequential(
            nn.Linear(dim // 4, dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 8, 2),
            nn.Softmax(dim=-1)
        )


    def forward(self, input_x):
        batch_size, channels, height, width = input_x.shape

        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)


        ca = self.out_CA(input_x)
        x = x.view(batch_size, height * width, -1)

        mask = self.out_mask(x)
        mask = F.gumbel_softmax(mask, hard=True, dim=2)[:, :, 0:1]
        #
        return mask, offsets, ca


class DiffCompletionSensor(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()

        self.dim = dim

        self.sub = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.cat = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)
        self.project_v = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.space =MSFD(dim)
        self.act = nn.GELU()

        self.route = Mask(dim)
        self.num_images = 0
    #

    def flow_warp(self, img, flow, interp_mode='bilinear', padding_mode='border'):

        B, C, H, W = img.size()


        grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_x, grid_y = grid_x.float().to(img.device), grid_y.float().to(img.device)

        grid_x = 2.0 * grid_x / (H - 1) - 1.0
        grid_y = 2.0 * grid_y / (W - 1) - 1.0
        grid = torch.stack((grid_y, grid_x), dim=-1)

        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        grid = grid + flow.permute(0, 2, 3, 1)

        warped_img = F.grid_sample(img, grid, mode=interp_mode, padding_mode=padding_mode)

        return warped_img



    def forward(self, t1, t2):
        a = []
        batch_size, channels, height, width = t1.shape

        sub = torch.abs(t1 - t2)
        cat = self.cat(torch.cat((t1, t2), 1))

        mask, offsets, ca = self.route(sub)
        v = cat
        q = cat
        k = cat + self.flow_warp(cat, offsets, interp_mode='bilinear', padding_mode='border')
        qk = torch.cat([q, k], dim=1)

        v1 = self.project_v(v)
        v1 = v1.view(batch_size, height * width, -1)
        qk = qk.view(batch_size, height * width, -1)

        qk1 = qk * mask

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)

        attn = torch.bmm(q1, k1.permute(0, 2, 1))
        attn = attn.softmax(dim=-1).permute(0, 2, 1)
        f_attn = torch.bmm(attn, v1)

        f_attn = f_attn.view(*t1.shape)
        out = f_attn * ca + cat

        out = self.space(out)
       
        return out


class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_redu = nn.Sequential(nn.Conv2d(dim * 2, dim, 1),
                                       nn.BatchNorm2d(dim, dim),
                                       nn.ReLU(inplace=True))


        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                   nn.BatchNorm2d(dim, dim),
                                   nn.ReLU(inplace=True))
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)
        output = self.conv_redu(output)
        res1 = output
        output = self.conv1(output) + res1

        return output




