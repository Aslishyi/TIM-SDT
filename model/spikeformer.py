from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *
from model.TIM import TIM

# class SpikeDrivenTransformer(nn.Module):
#     def __init__(
#         self,
#         img_size_h=128,
#         img_size_w=128,
#         patch_size=16,
#         in_channels=2,
#         num_classes=11,
#         embed_dims=512,
#         num_heads=8,
#         mlp_ratios=4,
#         qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.0,
#         norm_layer=nn.LayerNorm,
#         depths=[6, 8, 6],
#         sr_ratios=[8, 4, 2],
#         T=4,
#         pooling_stat="1111",
#         attn_mode="direct_xor",
#         spike_mode="lif",
#         get_embed=False,
#         dvs_mode=False,
#         TET=False,
#         cml=False,
#         pretrained=False,
#         pretrained_cfg=None,
#         TIM_alpha=0.6,
#         Dropout_prob=0.4
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths
#
#         self.T = T
#         self.TET = TET
#         self.dvs = dvs_mode
#         self.TIM = TIM(dim=embed_dims, TIM_alpha=TIM_alpha)
#
#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, depths)
#         ]  # stochastic depth decay rule
#
#         patch_embed = MS_SPS(
#             img_size_h=img_size_h,
#             img_size_w=img_size_w,
#             patch_size=patch_size,
#             in_channels=in_channels,
#             embed_dims=embed_dims,
#             pooling_stat=pooling_stat,
#             spike_mode=spike_mode,
#             dropout_prob=Dropout_prob
#         )
#
#         blocks = nn.ModuleList(
#             [
#                 MS_Block_Conv(
#                     dim=embed_dims,
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=dpr[j],
#                     norm_layer=norm_layer,
#                     sr_ratio=sr_ratios,
#                     attn_mode=attn_mode,
#                     spike_mode=spike_mode,
#                     dvs=dvs_mode,
#                     layer=j,
#                 )
#                 for j in range(depths)
#             ]
#         )
#
#         setattr(self, f"patch_embed", patch_embed)
#         setattr(self, f"block", blocks)
#
#         # classification head
#         if spike_mode in ["lif", "alif", "blif"]:
#             self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
#         elif spike_mode == "plif":
#             self.head_lif = MultiStepParametricLIFNode(
#                 init_tau=2.0, detach_reset=True, backend="cupy"
#             )
#         self.head = (
#             nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
#         )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward_features(self, x, hook=None):
#         block = getattr(self, f"block")
#         patch_embed = getattr(self, f"patch_embed")
#
#         x, _, hook = patch_embed(x, hook=hook)
#         for blk in block:
#             x, _, hook = blk(x, hook=hook)
#
#         x = x.flatten(3).mean(3)
#         return x, hook
#
#     def forward(self, x, hook=None):
#         if len(x.shape) == 4:
#             # Add temporal dimension if it's missing
#             x = x.unsqueeze(0)  # Adding temporal dimension T=1
#             x = x.repeat(self.T, 1, 1, 1, 1)
#         elif len(x.shape) == 5:
#             # If temporal dimension exists, just transpose
#             x = x.transpose(0, 1).contiguous()
#
#         x, hook = self.forward_features(x, hook=hook)
#         x = self.head_lif(x)
#         if hook is not None:
#             hook["head_lif"] = x.detach()
#
#         x = self.head(x)
#         if not self.TET:
#             x = x.mean(0)
#         return x, hook
class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],  # 默认值支持列表形式
        sr_ratios=[8, 4, 2],  # 默认值支持列表形式
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
        TIM_alpha=0.6,
        Dropout_prob=0.4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.TIM = TIM(dim=embed_dims, TIM_alpha=TIM_alpha)

        # 处理 depths 和 sr_ratios 的兼容性
        if isinstance(depths, int):
            total_depths = depths
            self.depths = [depths]  # 转换为单元素列表以保持一致性
        else:
            total_depths = sum(depths)
            self.depths = depths

        if not isinstance(sr_ratios, (list, tuple)):
            sr_ratios = [sr_ratios] * total_depths  # 如果是整数，扩展为列表
        elif len(sr_ratios) != total_depths:
            if len(sr_ratios) == 1:
                sr_ratios = sr_ratios * total_depths  # 单一值扩展
            else:
                raise ValueError(f"Length of sr_ratios ({len(sr_ratios)}) must match total depths ({total_depths})")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depths)]

        self.patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
            dropout_prob=Dropout_prob
        )

        self.blocks = nn.ModuleList([
            MS_Block_Conv(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[j],  # 使用对应层的 sr_ratio
                attn_mode=attn_mode,
                spike_mode=spike_mode,
                dvs=dvs_mode,
                layer=j
            )
            for j in range(total_depths)
        ])

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        x, _, hook = self.patch_embed(x, hook=hook)
        for blk in self.blocks:
            x, _, hook = blk(x, hook=hook)
        x = x.flatten(3).mean(3)  # Global Average Pooling
        return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape) == 5:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
        return x, hook

@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
