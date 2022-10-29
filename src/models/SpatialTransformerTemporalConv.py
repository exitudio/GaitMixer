# From PoseFormer: https://github.com/zczcwh/PoseFormer/edit/main/common/model_poseformer.py
# PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

from functools import partial
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.training:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_frame=31):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=.01)
        self.ref_pad = torch.nn.ReflectionPad2d((0, 0, kernel_frame-1, 0))
        # self.conv = torch.nn.Conv2d(in_channels=in_channels,
        #                         out_channels=out_channels,
        #                         kernel_size=(31, 1),
        #                         stride=1,
        #                         padding=0)
        self.conv = DepthwiseSeparableConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=(kernel_frame, 1),
                                           padding=0)
        self.act = torch.nn.GELU()
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.no_diff_c = False
        if in_channels != out_channels:
            self.linear = torch.nn.Linear(in_channels, out_channels)
            self.linear_act = torch.nn.GELU()
            self.bn_skip = torch.nn.BatchNorm2d(out_channels, eps=0.001)
            self.no_diff_c = True

    def forward(self, x):
        if self.no_diff_c:
            res_x = rearrange(x, 'b e f j -> b f j e')
            res_x = self.linear(res_x)
            res_x = rearrange(res_x, 'b f j e -> b e f j')
            res_x = self.linear_act(res_x)
            res_x = self.bn_skip(res_x)
        else:
            res_x = x

        x = self.dropout(x)
        x = self.ref_pad(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x += res_x
        return x


class SpatialTransformerTemporalConv(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=3, spatial_embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None, out_dim=124, kernel_frame=31):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.spatial_embed_dim = spatial_embed_dim
        self.final_embed_dim = 256 #spatial_embed_dim #8*spatial_embed_dim # spatial_embed_dim * num_joints
        self.out_dim = out_dim

        # spatial patch embedding
        self.spatial_joint_to_embedding = nn.Linear(
            in_chans, spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, spatial_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.spatial_norm = norm_layer(spatial_embed_dim)

        self.conv1 = ConvBlock(
            in_channels=32, out_channels=32, kernel_frame=kernel_frame)
        self.conv2 = ConvBlock(
            in_channels=32, out_channels=64, kernel_frame=kernel_frame)
        self.conv3 = ConvBlock(
            in_channels=64, out_channels=128, kernel_frame=kernel_frame)
        self.conv4 = ConvBlock(
            in_channels=128, out_channels=self.final_embed_dim, kernel_frame=kernel_frame)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(num_frame, num_joints), stride=1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.final_embed_dim),
            nn.Linear(self.final_embed_dim, self.out_dim),
        )

    def spatial_transformer(self, x):
        for blk in self.spatial_blocks:
            x = blk(x)
        return self.spatial_norm(x)

    def spatial_forward(self, x):
        b, f, j, d = x.shape
        x = rearrange(x, 'b f j d -> (b f) j  d')
        x = self.spatial_joint_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) j e -> b e f j', f=f)
        return x

    def temporal_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        return x

    def forward(self, x):
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        x = self.head(x)
        x = F.normalize(x, dim=1, p=2)
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=3, spatial_embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None, out_dim=124):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.spatial_embed_dim = spatial_embed_dim
        self.temporal_embed_dim = spatial_embed_dim #* num_joints
        self.out_dim = out_dim 

        # spatial patch embedding
        self.spatial_joint_to_embedding = nn.Linear(
            in_chans, spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, spatial_embed_dim))

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frame+1, self.temporal_embed_dim))
        self.temporal_cls_token = nn.Parameter(
            torch.randn(1, 1, self.temporal_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.temporal_blocks = nn.ModuleList([
            Block(
                dim=self.temporal_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.spatial_norm = norm_layer(spatial_embed_dim)
        self.temporal_norm = norm_layer(self.temporal_embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(self.temporal_embed_dim* num_joints),
            nn.Linear(self.temporal_embed_dim* num_joints, self.out_dim),
        )

    def spatial_transformer(self, x):
        for blk in self.spatial_blocks:
            x = blk(x)
        return self.spatial_norm(x)

    def temporal_transformer(self, x):
        for blk in self.temporal_blocks:
            x = blk(x)
        return self.temporal_norm(x)

    def spatial_forward(self, x):
        b, f, j, d = x.shape
        x = rearrange(x, 'b f j d -> (b f) j  d')
        x = self.spatial_joint_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) j e -> (b j) f e', f=f)
        return x

    def temporal_forward(self, x):
        b, f, e = x.shape
        cls_tokens = repeat(self.temporal_cls_token, '() 1 e -> b 1 e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)

        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b j) f e -> b f (j e)', j=17)
        x = x[:, 0, :]
        return x

    def forward(self, x):
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        x = self.head(x)
        x = F.normalize(x, dim=1, p=2)
        return x