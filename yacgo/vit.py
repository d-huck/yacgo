""" EfficientFormer-V2

@article{
    li2022rethinking,
    title={Rethinking Vision Transformers for MobileNet Size and Speed},
    author={Li, Yanyu and Hu, Ju and Wen, Yang and Evangelidis, Georgios and 
        Salahi, Kamyar and Wang, Yanzhi and Tulyakov, Sergey and Ren, Jian},
    journal={arXiv preprint arXiv:2212.08059},
    year={2022}
}

Significantly refactored and cleaned up for timm from original at:
https://github.com/snap-research/EfficientFormer

Original code licensed Apache 2.0, Copyright (c) 2022 Snap Inc.

Modifications and timm support by / Copyright 2023, Ross Wightman
"""

# pylint: disable=line-too-long,missing-class-docstring,missing-function-docstring,missing-module-docstring,invalid-name

import math
from functools import partial
from typing import Dict
import numpy as np

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    create_conv2d,
    create_norm_layer,
    get_act_layer,
    get_norm_layer,
    ConvNormAct,
)
from timm.layers import DropPath, trunc_normal_, to_2tuple, to_ntuple, ndgrid
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import generate_default_cfgs

torch.autograd.set_detect_anomaly(True)

EfficientFormer_width = {
    "L": (40, 80, 192, 384),  # 26m 83.3% 6attn
    "S2": (32, 64, 144, 288),  # 12m 81.6% 4attn dp0.02
    "S1": (32, 48, 120, 224),  # 6.1m 79.0
    "S0": (32, 48, 96, 176),  # 75.0 75.7
}

EfficientFormer_depth = {
    "L": (5, 5, 15, 10),  # 26m 83.3%
    "S2": (4, 4, 12, 8),  # 12m
    "S1": (3, 3, 9, 6),  # 79.0
    "S0": (2, 2, 6, 4),  # 75.7
}

EfficientFormer_expansion_ratios = {
    "L": (
        4,
        4,
        (4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4),
        (4, 4, 4, 3, 3, 3, 3, 4, 4, 4),
    ),
    "S2": (4, 4, (4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4), (4, 4, 3, 3, 3, 3, 4, 4)),
    "S1": (4, 4, (4, 4, 3, 3, 3, 3, 4, 4, 4), (4, 4, 3, 3, 4, 4)),
    "S0": (4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
}


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=True,
        norm_layer="batchnorm2d",
        norm_kwargs=None,
    ):
        norm_kwargs = norm_kwargs or {}
        super(ConvNorm, self).__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = create_norm_layer(norm_layer, out_channels, **norm_kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Attention2d(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
        self,
        dim=384,
        key_dim=32,
        num_heads=8,
        attn_ratio=4,
        resolution=7,
        act_layer=nn.GELU,
        stride=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim

        resolution = to_2tuple(resolution)
        orig_resolution = resolution
        if stride is not None:
            resolution = tuple([math.ceil(r / stride) for r in resolution])
            self.stride_conv = ConvNorm(
                dim, dim, kernel_size=3, stride=stride, groups=dim
            )
            self.upsample = nn.Upsample(size=orig_resolution, mode="bilinear")
        else:
            self.stride_conv = None
            self.upsample = None

        self.resolution = resolution
        self.N = self.resolution[0] * self.resolution[1]
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        kh = self.key_dim * self.num_heads

        self.q = ConvNorm(dim, kh)
        self.k = ConvNorm(dim, kh)
        self.v = ConvNorm(dim, self.dh)
        self.v_local = ConvNorm(self.dh, self.dh, kernel_size=3, groups=self.dh)
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        self.act = act_layer()
        self.proj = ConvNorm(self.dh, dim, 1)

        pos = torch.stack(
            ndgrid(torch.arange(self.resolution[0]), torch.arange(self.resolution[1]))
        ).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, self.N))
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(rel_pos), persistent=False
        )
        self.attention_bias_cache = (
            {}
        )  # per-device attention_biases cache (data-parallel compat)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[
                    :, self.attention_bias_idxs
                ]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        B, C, H, W = x.shape  # pylint: disable=unused-variable
        if self.stride_conv is not None:
            x = self.stride_conv(x)
        q = self.q(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn + self.get_attention_biases(x.device)
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v).transpose(2, 3)
        x = x.reshape(B, self.dh, self.resolution[0], self.resolution[1]) + v_local
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.act(x)
        x = self.proj(x)
        return x


class LocalGlobalQuery(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim
        )
        self.proj = ConvNorm(in_dim, out_dim, 1)

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention2dDownsample(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
        self,
        dim=384,
        key_dim=16,
        num_heads=8,
        attn_ratio=4,
        resolution=7,
        out_dim=None,
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.resolution = to_2tuple(resolution)
        self.resolution2 = tuple([math.ceil(r / 2) for r in self.resolution])
        self.N = self.resolution[0] * self.resolution[1]
        self.N2 = self.resolution2[0] * self.resolution2[1]

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.out_dim = out_dim or dim
        kh = self.key_dim * self.num_heads

        self.q = LocalGlobalQuery(dim, kh)
        self.k = ConvNorm(dim, kh, 1)
        self.v = ConvNorm(dim, self.dh, 1)
        self.v_local = ConvNorm(
            self.dh, self.dh, kernel_size=3, stride=2, groups=self.dh
        )

        self.act = act_layer()
        self.proj = ConvNorm(self.dh, self.out_dim, 1)

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, self.N))
        k_pos = torch.stack(
            ndgrid(torch.arange(self.resolution[0]), torch.arange(self.resolution[1]))
        ).flatten(1)
        q_pos = torch.stack(
            ndgrid(
                torch.arange(0, self.resolution[0], step=2),
                torch.arange(0, self.resolution[1], step=2),
            )
        ).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        self.register_buffer("attention_bias_idxs", rel_pos, persistent=False)
        self.attention_bias_cache = (
            {}
        )  # per-device attention_biases cache (data-parallel compat)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[
                    :, self.attention_bias_idxs
                ]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        B, C, H, W = x.shape  # pylint: disable=unused-variable

        q = self.q(x).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn + self.get_attention_biases(x.device)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(2, 3)
        x = x.reshape(B, self.dh, self.resolution2[0], self.resolution2[1]) + v_local
        x = self.act(x)
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=3,
        stride=2,
        padding=1,
        resolution=7,
        use_attn=False,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        padding = kernel_size - 1
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        resolution = to_2tuple(resolution)
        norm_layer = norm_layer or nn.Identity()
        self.conv = ConvNorm(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
        )

        if use_attn:
            self.attn = Attention2dDownsample(
                dim=in_chs,
                out_dim=out_chs,
                resolution=resolution,
                act_layer=act_layer,
            )
        else:
            self.attn = None
        self.upsample = nn.Upsample(size=resolution, mode="bilinear")

    def forward(self, x):
        out = self.conv(x)
        if self.attn is not None:
            return self.attn(x) + out
        return self.upsample(out)


class ConvMlpWithNorm(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        drop=0.0,
        mid_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvNormAct(
            in_features,
            hidden_features,
            1,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        if mid_conv:
            self.mid = ConvNormAct(
                hidden_features,
                hidden_features,
                3,
                groups=hidden_features,
                bias=True,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.mid = nn.Identity()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = ConvNorm(hidden_features, out_features, 1, norm_layer=norm_layer)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.mid(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class EfficientFormerV2Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
        resolution=7,
        stride=None,
        use_attn=True,
    ):
        super().__init__()
        if use_attn:
            self.token_mixer = Attention2d(
                dim,
                resolution=resolution,
                act_layer=act_layer,
                stride=stride,
            )
            self.ls1 = (
                LayerScale2d(dim, layer_scale_init_value)
                if layer_scale_init_value is not None
                else nn.Identity()
            )
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        else:
            self.token_mixer = None
            self.ls1 = None
            self.drop_path1 = None
        # mid_conv = not use_attn
        self.mlp = ConvMlpWithNorm(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=proj_drop,
            mid_conv=True,
        )
        self.ls2 = (
            LayerScale2d(dim, layer_scale_init_value)
            if layer_scale_init_value is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.token_mixer is not None:
            x_ = self.token_mixer(x)
            x_ = self.ls1(x_)
            x = x + self.drop_path1(x_)
            # x = x + self.drop_path1(self.ls1(self.token_mixer(x)))
        x_ = self.mlp(x)
        x_ = self.ls2(x_)
        x = x + self.drop_path2(x_)
        # x = x + self.drop_path2(self.ls2(self.mlp(x)))
        return x


class Stem(nn.Sequential):
    def __init__(
        self, in_chs, out_chs, stride=4, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.stride = stride
        if stride == 1:
            stride1 = 1
            stride2 = 1
        else:
            stride1 = stride // 2
            stride2 = stride // 2 + stride % 2
        padding1 = stride1 - 1
        padding2 = stride2 - 1
        self.stride = 4
        self.conv1 = ConvNormAct(
            in_chs,
            out_chs // 2,
            kernel_size=3,
            stride=stride1,
            padding=1,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.conv2 = ConvNormAct(
            out_chs // 2,
            out_chs,
            kernel_size=3,
            stride=stride2,
            padding=1,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )


class EfficientFormerV2Stage(nn.Module):

    def __init__(
        self,
        dim,
        dim_out,
        depth,
        resolution=7,
        downsample=True,
        block_stride=None,
        downsample_use_attn=False,
        block_use_attn=False,
        num_vit=4,
        mlp_ratio=4.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.grad_checkpointing = False
        mlp_ratio = to_ntuple(depth)(mlp_ratio)
        resolution = to_2tuple(resolution)
        if downsample:
            self.downsample = Downsample(
                dim,
                dim_out,
                use_attn=downsample_use_attn,
                resolution=resolution,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            dim = dim_out
            # resolution = tuple([math.ceil(r / 2) for r in resolution])
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            remain_idx = depth - num_vit - 1
            b = EfficientFormerV2Block(
                dim,
                resolution=resolution,
                stride=block_stride,
                mlp_ratio=mlp_ratio[block_idx],
                use_attn=block_use_attn and block_idx > remain_idx,
                proj_drop=proj_drop,
                drop_path=drop_path[block_idx],
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            blocks += [b]
        self.blocks = nn.Sequential(*blocks)
        self.upsample = nn.Conv2d(dim, dim_out, 1, stride=3, padding=2)
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class ValueHead(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        global_pool="avg",
        drop_rate=0.0,
    ):
        super().__init__()
        self.global_pool = global_pool
        linear_in = num_classes - 1
        self.head = nn.Sequential(
            nn.Conv2d(num_features, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(linear_in, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.head(x)


class PolicyHead(nn.Module):
    def __init__(
        self,
        num_features,
        num_actions,
        drop_rate=0.0,
        global_pool="avg",
        distillation=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.head_drop = nn.Dropout(drop_rate)
        self.num_actions = num_actions
        self.global_pool = global_pool
        self.dist = distillation
        board_area = (num_actions - 1) * 2
        self.head = nn.Sequential(
            nn.Conv2d(num_features, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(board_area, num_actions),
        )
        self.head_dist = (
            nn.Linear(num_features, num_actions)
            if num_actions > 0 and distillation
            else nn.Identity()
        )

        self.distilled_training = False

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward(self, x, pre_logits: bool = False):
        x = self.head_drop(x)
        if pre_logits:
            return x
        return self.head(x)


class EfficientFormerV2(nn.Module):
    def __init__(
        self,
        depths,
        in_chans=3,
        img_size=224,
        global_pool="avg",
        embed_dims=None,
        downsamples=None,
        mlp_ratios=4,
        norm_layer="batchnorm2d",
        norm_eps=1e-5,
        act_layer="gelu",
        num_classes=1000,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        num_vit=0,
        distillation=True,
    ):
        super().__init__()
        assert global_pool in ("avg", "")
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_info = []
        img_size = to_2tuple(img_size)
        norm_layer = partial(get_norm_layer(norm_layer), eps=norm_eps)
        act_layer = get_act_layer(act_layer)

        stride = 1
        self.stem = Stem(
            in_chans,
            embed_dims[0],
            stride=stride,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        prev_dim = embed_dims[0]

        num_stages = len(depths)
        dpr = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        downsamples = downsamples or (False,) + (True,) * (len(depths) - 1)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        stages = []
        for i in range(num_stages):
            # curr_resolution = tuple([math.ceil(s / stride) for s in img_size])
            stage = EfficientFormerV2Stage(
                prev_dim,
                embed_dims[i],
                depth=depths[i],
                resolution=img_size,
                downsample=downsamples[i],
                block_stride=2 if i == 2 else None,
                downsample_use_attn=False,  # i >= 3,
                block_use_attn=i >= 2,
                num_vit=num_vit,
                mlp_ratio=mlp_ratios[i],
                proj_drop=proj_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            if downsamples[i]:
                stride *= 2
            prev_dim = embed_dims[i]
            self.feature_info += [
                dict(num_chs=prev_dim, reduction=stride, module=f"stages.{i}")
            ]
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        # Policy and value heads
        self.dist = distillation
        self.num_features = embed_dims[-1]
        self.norm = norm_layer(embed_dims[-1])

        self.policy_head = PolicyHead(
            embed_dims[-1],
            num_classes,
            drop_rate=drop_rate,
            global_pool=global_pool,
            distillation=self.dist,
        )

        self.value_head = ValueHead(
            embed_dims[-1],
            num_classes,
            drop_rate=drop_rate,
            global_pool=global_pool,
        )

        self.apply(self.init_weights)
        self.distilled_training = False

    # init for classification
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters() if "attention_biases" in k}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):  # pylint: disable=unused-argument
        matcher = dict(
            stem=r"^stem",  # stem and embed
            blocks=[(r"^stages\.(\d+)", None), (r"^norm", (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.policy_head = PolicyHead(
            self.num_features,
            num_classes,
            drop_rate=self.drop_rate,
            global_pool=global_pool,
            distillation=self.dist,
        )

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable
        self.policy_head.set_distilled_training(enable)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        value = self.value_head(x)
        policy = self.policy_head(x, pre_logits)
        return value, policy

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)

    def forward_state(self, state):
        value, policy = self(torch.from_numpy(state).float().unsqueeze(0))
        return value.item(), policy.squeeze().detach().numpy()
        # return np.random.random(), np.random.random(26)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "fixed_input_size": True,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": ("head", "head_dist"),
        "first_conv": "stem.conv1.conv",
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "efficientformerv2_s0.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
        "efficientformerv2_s1.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
        "efficientformerv2_s2.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
        "efficientformerv2_l.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
    }
)


def _create_efficientformerv2(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop("out_indices", (0, 1, 2, 3))
    model = build_model_with_cfg(
        EfficientFormerV2,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model
