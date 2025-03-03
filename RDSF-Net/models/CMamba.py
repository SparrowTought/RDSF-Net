import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from pytorch_wavelets import DWTForward
import math
from .SSM import SSM
from einops import rearrange, repeat
import matplotlib.pyplot as plt


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

        out = identity * a_w * a_h

        return out
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')


    def forward(self, x):
        yL, yH = self.wt(x)  # Apply DWT
        y_HL = yH[0][:,:,0,:,:]
        y_LH = yH[0][:,:,1,:,:]
        y_HH = yH[0][:,:,2,:,:]
        # x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # Concatenate subbands
        # x = self.conv_bn_relu(x)

        return yL, y_HL, y_LH, y_HH

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        self.in_channels = in_channels

        # Sobel filter for edge detection in the X direction
        self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        # Sobel filter for edge detection in the Y direction
        self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

        self.sobel_xy = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

        self.sobel_yx = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.initialize_weights()

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_xy = self.sobel_xy(x)
        edge_yx = self.sobel_yx(x)
        # Combine edges in both directions
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + edge_xy ** 2 + edge_yx ** 2) * self.gamma
        return edge

    def initialize_weights(self):
        """Initialize the weights of the convolutional layers with Sobel kernels."""
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                               3)
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                               3)
        sobel_kernel_xy = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                               3)
        sobel_kernel_yx = torch.tensor([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                               3)

        # Repeat the Sobel kernel for each input channel
        sobel_kernel_x = sobel_kernel_x.repeat(self.in_channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(self.in_channels, 1, 1, 1)
        sobel_kernel_xy = sobel_kernel_xy.repeat(self.in_channels, 1, 1, 1)
        sobel_kernel_yx = sobel_kernel_yx.repeat(self.in_channels, 1, 1, 1)

        with torch.no_grad():
            self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
            self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)
            self.sobel_xy.weight = nn.Parameter(sobel_kernel_xy, requires_grad=False)
            self.sobel_yx.weight = nn.Parameter(sobel_kernel_yx, requires_grad=False)

class space(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                                   nn.Conv2d(dim, dim, 1),
                                   nn.BatchNorm2d(dim),
                                    nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                                   nn.Conv2d(dim, dim, 1),
                                   nn.BatchNorm2d(dim),
                                    nn.ReLU(inplace=True))

        self.ca = CoordAtt(dim)

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.conv2(x1)

        out = self.ca(x1)

        return out
class Conv(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.Canny = space(dim)
    def forward(self,  x):

        x = x.permute(0, 3, 1, 2) # BCHW

        out1 = self.Canny(x)

        out = out1.permute(0, 3, 2, 1)# BHWC
        return out


class Block(nn.Module):

    def __init__(self, dim,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 depths=0,
                 id_stage=0,
                 nums=0,
                ssm_d_state=16,
                ssm_ratio = 2.0,
                ssm_dt_rank = "auto",
                ssm_act_layer = nn.SiLU,
                ssm_conv: int = 3,
                ssm_conv_bias = True,
                ssm_drop_rate=0.0,
                ssm_init = "v0",
                forward_type = "v2",
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.depth = depths
        self.Conv = Conv(dim)
        self.drop_path = drop_path
        # self.short = Short(dim, dim)
        self.id_stage = id_stage
        self.num = nums
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0



        if self.ssm_branch:
            self.norm = norm_layer(dim)
            self.op = SSM(
                d_model=dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=dim//2, act_layer=nn.GELU, channels_first=False)
        self.weight_net = nn.Conv2d(dim, dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        op = self.drop_path(self.op(self.norm1(x)))

        att_outputs = op + x

        att_outputs = att_outputs + self.drop_path(self.mlp(self.norm2(att_outputs)))

        return att_outputs

class edge(nn.Module):
    def __init__(self, in_channels):
        super(edge, self).__init__()
        self.in_channels = in_channels
        # Sobel filter for edge detection in the X direction
        self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        # Sobel filter for edge detection in the Y direction
        self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.initialize_weights()

    def forward(self, HL, LH):
        edge_x = self.sobel_x(HL)
        edge_y = self.sobel_y(LH)
        # Combine edges in both directions
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge

    def initialize_weights(self):
        """Initialize the weights of the convolutional layers with Sobel kernels."""
        sobel_kernel_x = torch.tensor([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                               3)
        sobel_kernel_y = torch.tensor([[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                               3)
        # Repeat the Sobel kernel for each input channel
        sobel_kernel_x = sobel_kernel_x.repeat(self.in_channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(self.in_channels, 1, 1, 1)

        with torch.no_grad():
            self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
            self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

class make_downsample_v4(nn.Module):
    def __init__(self, dim=96, out_dim=192, norm_layer=nn.LayerNorm, deep=0):
        super(make_downsample_v4, self).__init__()
        self.dw = Down_wt(dim, dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(out_dim),
            # nn.ReLU(inplace=True),
        )

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(dim * 4, out_dim, kernel_size=3, stride=1, padding=1),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(out_dim),
            # nn.ReLU(inplace=True),
        )



    def forward(self, x1, x2):
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        c1 = self.conv(x1)
        c2 = self.conv(x2)
        yL1, y_HL1, y_LH1, y_HH1 = self.dw(x1)
        yL2, y_HL2, y_LH2, y_HH2 = self.dw(x2)

        x1 = torch.cat([yL1, y_HL1, y_LH1, y_HH1], dim=1)  # Concatenate subbands
        x1 = self.conv_bn_relu(x1) + c1
        x2 = torch.cat([yL2, y_HL2, y_LH2, y_HH2], dim=1)  # Concatenate subbands
        x2 = self.conv_bn_relu(x2) + c2

        return x1, x2



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        out = self.sigmoid(avg_out)
        return out


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class CMamba(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths


        def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
            assert patch_size == 4
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
                (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
                (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                Permute(0, 2, 3, 1),
                (norm_layer(embed_dim) if patch_norm else nn.Identity()),
            )

        def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
            return nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                Permute(0, 2, 3, 1),
                norm_layer(out_dim),
            )
        self.patch_embed2 = make_downsample_v4(dim=embed_dims[0],
                                              out_dim=embed_dims[1])
        self.patch_embed3 = make_downsample_v4(dim=embed_dims[1],
                                              out_dim=embed_dims[2])
        self.patch_embed4 = make_downsample_v4(dim=embed_dims[2],
                                              out_dim=embed_dims[3])

        # patch_embed
        self.patch_embed1 = _make_patch_embed_v2(in_chans=3,embed_dim=embed_dims[0],patch_size=4,norm_layer=nn.LayerNorm
                                              )


        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0],  mlp_ratio=mlp_ratios[0],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
         id_stage=i, nums=0, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,ssm_conv=3,ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0",forward_type="v2",)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1],  mlp_ratio=mlp_ratios[1],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
             id_stage=i, nums=1, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,ssm_conv=3,ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0",forward_type="v2",)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], mlp_ratio=mlp_ratios[2],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
             id_stage=i, nums=2, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,ssm_conv=3,ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0",forward_type="v2",)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], mlp_ratio=mlp_ratios[3],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            id_stage=i, nums=3, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,ssm_conv=3,ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0",forward_type="v2",)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])


        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



    def forward_features(self, x1, x2):
        B = x1.shape[0]
        outs1 = []
        outs2 = []

        # stage 1
        x1= self.patch_embed1(x1)
        x2= self.patch_embed1(x2)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1)
            x2 = blk(x2)
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        x2_1 = rearrange(x2, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)
        outs2.append(x2_1)

        # stage 2
        x1, x2 = self.patch_embed2(x1, x2)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1)
            x2 = blk(x2)
        x1 = self.norm2(x1)
        x2 = self.norm2(x2)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        x2_1 = rearrange(x2, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)
        outs2.append(x2_1)

        # stage 3
        x1, x2 = self.patch_embed3(x1, x2)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1)
            x2 = blk(x2)
        x1 = self.norm3(x1)
        x2 = self.norm3(x2)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        x2_1 = rearrange(x2, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)
        outs2.append(x2_1)

        x1, x2 = self.patch_embed4(x1, x2)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1)
            x2 = blk(x2)
        x1 = self.norm4(x1)
        x2 = self.norm4(x2)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        x2_1 = rearrange(x2, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)
        outs2.append(x2_1)


        return outs1, outs2

        # return x.mean(dim=1)

    def forward(self, x1, x2):
        x1, x2 = self.forward_features(x1, x2)
        # x = self.head(x)

        return x1, x2


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
class CMamba(CMamba):
    def __init__(self, **kwargs):
        super(CMamba, self).__init__(
            patch_size=4, embed_dims=[64, 96, 192, 384], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 12, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

