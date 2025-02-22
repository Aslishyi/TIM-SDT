from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from module import MS_SPS, MS_Block_Conv
from model.TIM import TIM

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
        depths=2,
        sr_ratios=[1, 1],
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

        if isinstance(sr_ratios, int):
            sr_ratios = [sr_ratios] * 3
        elif len(sr_ratios) < 3:
            sr_ratios = sr_ratios + [sr_ratios[-1]] * (3 - len(sr_ratios))
        self.sr_ratios = sr_ratios

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            self._get_lif(spike_mode),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage_dims = [64, 192, embed_dims]
        self.stages = nn.ModuleList()
        total_depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depths)]
        current_depth = 0

        # Stage 1
        stage1_blocks = nn.ModuleList([
            MS_Block_Conv(
                dim=self.stage_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[current_depth + j],
                norm_layer=norm_layer, sr_ratio=self.sr_ratios[0], attn_mode=attn_mode, spike_mode=spike_mode,
                dvs=dvs_mode, layer=j
            ) for j in range(total_depths // 3)
        ])
        self.stages.append(stage1_blocks)
        current_depth += total_depths // 3

        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.stage_dims[0], self.stage_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_dims[1]),
            self._get_lif(spike_mode)
        )

        # Stage 2
        stage2_blocks = nn.ModuleList([
            MS_Block_Conv(
                dim=self.stage_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[current_depth + j],
                norm_layer=norm_layer, sr_ratio=self.sr_ratios[1], attn_mode=attn_mode, spike_mode=spike_mode,
                dvs=dvs_mode, layer=j + total_depths // 3
            ) for j in range(total_depths // 3)
        ])
        self.stages.append(stage2_blocks)
        current_depth += total_depths // 3

        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.stage_dims[1], self.stage_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_dims[2]),
            self._get_lif(spike_mode)
        )

        # Stage 3
        stage3_blocks = nn.ModuleList([
            MS_Block_Conv(
                dim=self.stage_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[current_depth + j],
                norm_layer=norm_layer, sr_ratio=self.sr_ratios[2], attn_mode=attn_mode, spike_mode=spike_mode,
                dvs=dvs_mode, layer=j + 2 * (total_depths // 3)
            ) for j in range(total_depths - 2 * (total_depths // 3))
        ])
        self.stages.append(stage3_blocks)

        self.patch_embed = MS_SPS(
            img_size_h=img_size_h, img_size_w=img_size_w, patch_size=patch_size,
            in_channels=self.stage_dims[-1], embed_dims=embed_dims, pooling_stat=pooling_stat,
            spike_mode=spike_mode, dropout_prob=Dropout_prob
        )

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _get_lif(self, spike_mode, v_threshold=1.0):
        if spike_mode == "lif":
            return MultiStepLIFNode(tau=2.0, v_threshold=v_threshold, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            return MultiStepParametricLIFNode(init_tau=2.0, v_threshold=v_threshold, detach_reset=True, backend="cupy")
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        T, B, C, H, W = x.shape
        # print(f"Input x: shape={x.shape}, min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        x = x.view(T * B, C, H, W)
        x = self.stem(x)
        _, C_new, H_new, W_new = x.shape
        x = x.view(T, B, C_new, H_new, W_new).contiguous()

        for stage in self.stages:
            for blk in stage:
                x, _, hook = blk(x, hook=hook)
            if stage is self.stages[0]:
                x = x.view(T * B, x.shape[2], x.shape[3], x.shape[4])
                x = self.downsample1(x)
                x = x.view(T, B, self.stage_dims[1], x.shape[2], x.shape[3]).contiguous()
            elif stage is self.stages[1]:
                x = x.view(T * B, x.shape[2], x.shape[3], x.shape[4])
                x = self.downsample2(x)
                x = x.view(T, B, self.stage_dims[2], x.shape[2], x.shape[3]).contiguous()
        x, _, hook = self.patch_embed(x, hook=hook)
        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        elif len(x.shape) == 5:
            x = x.transpose(0, 1).contiguous()  # [T, B, C, H, W]

        x, hook = self.forward_features(x, hook=hook)
        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
        return x, hook

@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(**kwargs)
    model.default_cfg = _cfg()
    return model