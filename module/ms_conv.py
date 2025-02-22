import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from model.TIM import TIM

class Erode(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)

class DSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0,
                 proj_drop=0.0, sr_ratio=1, spike_mode="lif", dvs=False, layer=0, TIM_alpha=0.6):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dvs = dvs
        self.sr_ratio = sr_ratio
        self.spike_mode = spike_mode  # 保存 spike_mode 用于动态创建 LIF
        self.TIM = TIM(dim=dim, in_channels=dim, TIM_alpha=TIM_alpha)
        if dvs:
            self.pool = Erode()
        self.scale = 0.125

        # QKV convolutions
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)

        # Projection layer
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        # DST convolution for spatial reduction
        self.dst_conv = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=False)
        self.dst_bn = nn.BatchNorm2d(self.head_dim)

        self.layer = layer

        # 初始化权重
        nn.init.kaiming_normal_(self.q_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.k_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.v_conv.weight, mode='fan_out', nonlinearity='relu')

    def _get_lif(self, v_threshold=1.0):
        if self.spike_mode == "lif":
            return MultiStepLIFNode(tau=2.0, v_threshold=v_threshold, detach_reset=True, backend="cupy")
        elif self.spike_mode == "plif":
            return MultiStepParametricLIFNode(init_tau=2.0, v_threshold=v_threshold, detach_reset=True, backend="cupy")
        raise ValueError(f"Unsupported spike_mode: {self.spike_mode}")

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        # 动态创建所有 LIF 神经元
        shortcut_lif = self._get_lif()
        q_lif = self._get_lif(v_threshold=0.05)
        k_lif = self._get_lif()
        v_lif = self._get_lif()
        attn_lif = self._get_lif(v_threshold=0.5)
        proj_lif = self._get_lif()

        x = shortcut_lif(x)
        if hook is not None:
            hook[f"{self._get_name()}{self.layer}_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        # print(f"x_for_qkv: min={x_for_qkv.min().item()}, max={x_for_qkv.max().item()}, mean={x_for_qkv.mean().item()}")
        q_conv_out = self.q_conv(x_for_qkv)
        # print(f"q_conv_out: min={q_conv_out.min().item()}, max={q_conv_out.max().item()}, mean={q_conv_out.mean().item()}")
        q = q_conv_out.reshape(T, B, C, H, W).contiguous()
        # print(f"q before q_lif: min={q.min().item()}, max={q.max().item()}, mean={q.mean().item()}")
        q = q_lif(q)
        # print(f"q after q_lif: min={q.min().item()}, max={q.max().item()}, mean={q.mean().item()}")
        q_tim_in = q.view(T, B, -1, H, W)
        q = self.TIM(q_tim_in)
        # print(f"TIM input: min={q_tim_in.min().item()}, max={q_tim_in.max().item()}, mean={q_tim_in.mean().item()}")
        # print(f"q after TIM: min={q.min().item()}, max={q.max().item()}, mean={q.mean().item()}")
        q = q.view(T, B, C, H, W)

        k = k_lif(self.k_bn(self.k_conv(x_for_qkv)).reshape(T, B, C, H, W).contiguous())
        if self.dvs:
            k = self.pool(k)
        if hook is not None:
            hook[f"{self._get_name()}{self.layer}_k_lif"] = k.detach()

        v = v_lif(self.v_bn(self.v_conv(x_for_qkv)).reshape(T, B, C, H, W).contiguous())
        if self.dvs:
            v = self.pool(v)
        if hook is not None:
            hook[f"{self._get_name()}{self.layer}_v_lif"] = v.detach()

        head_dim = C // self.num_heads
        q = q.view(T, B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        k = k.view(T, B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        v = v.view(T, B, self.num_heads, head_dim, H * W).transpose(-1, -2)

        k_flat = k.reshape(T * B * self.num_heads, head_dim, H, W)
        k_transformed = self.dst_bn(self.dst_conv(k_flat))
        H_prime, W_prime = k_transformed.shape[-2], k_transformed.shape[-1]
        k_transformed = k_transformed.view(T * B * self.num_heads, head_dim, H_prime * W_prime).transpose(-1, -2)
        k_transformed = k_transformed.view(T, B, self.num_heads, H_prime * W_prime, head_dim)

        attn_input = q @ k_transformed.transpose(-1, -2)
        firing_rate_x = q.mean()
        c1 = 1 / torch.sqrt(firing_rate_x * head_dim + 1e-6)
        c1 = torch.clamp(c1, max=10.0)
        attn_input_scaled = attn_input * c1
        # print(f"attn_input_scaled: min={attn_input_scaled.min().item()}, max={attn_input_scaled.max().item()}")

        attn_input_scaled = attn_input_scaled.view(T, B * self.num_heads * H * W, H_prime * W_prime)
        attn_map = attn_lif(attn_input_scaled)
        attn_map = attn_map.view(T, B, self.num_heads, H * W, H_prime * W_prime)

        v_flat = v.reshape(T * B * self.num_heads, head_dim, H, W)
        v_transformed = self.dst_bn(self.dst_conv(v_flat))
        v_transformed = v_transformed.view(T * B * self.num_heads, head_dim, H_prime * W_prime).transpose(-1, -2)
        v_transformed = v_transformed.view(T, B, self.num_heads, H_prime * W_prime, head_dim)

        out = attn_map @ v_transformed
        firing_rate_attn = attn_map.mean(dim=(0, 1, 3, 4))
        c2 = 1 / torch.sqrt(firing_rate_attn * (H * W) / (self.dst_conv.stride[0] ** 2) + 1e-6)

        if self.sr_ratio > 1:
            out = out.transpose(-1, -2).reshape(T, B, C, H_prime, W_prime)
            identity = F.interpolate(identity.flatten(0, 1), size=(H_prime, W_prime), mode='bilinear', align_corners=False)
            identity = identity.view(T, B, C, H_prime, W_prime)
        else:
            out = out.transpose(-1, -2).reshape(T, B, C, H, W)

        # print(f"out shape before proj_lif: {out.shape}")
        # 先加残差，后通过 ConvBN 和 LIF
        out = out + identity
        out = proj_lif(self.proj_bn(self.proj_conv(out.flatten(0, 1))).reshape(T, B, C, H_prime if self.sr_ratio > 1 else H, W_prime if self.sr_ratio > 1 else W).contiguous())

        return out, v, hook

class MS_MLP_Conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, spike_mode="lif", layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.spike_mode = spike_mode

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def _get_lif(self, v_threshold=1.0):
        if self.spike_mode == "lif":
            return MultiStepLIFNode(tau=2.0, v_threshold=v_threshold, detach_reset=True, backend="cupy")
        elif self.spike_mode == "plif":
            return MultiStepParametricLIFNode(init_tau=2.0, v_threshold=v_threshold, detach_reset=True, backend="cupy")
        raise ValueError(f"Unsupported spike_mode: {self.spike_mode}")

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        fc1_lif = self._get_lif()
        fc2_lif = self._get_lif()

        x = fc1_lif(x)
        if hook is not None:
            hook[f"{self._get_name()}{self.layer}_fc1_lif"] = x.detach()
        x = self.fc1_bn(self.fc1_conv(x.flatten(0, 1))).reshape(T, B, self.c_hidden, H, W).contiguous()

        if self.res:
            x = x + identity
        x = fc2_lif(self.fc2_bn(self.fc2_conv(x.flatten(0, 1))).reshape(T, B, self.c_output, H, W).contiguous())

        return x, hook

class MS_Block_Conv(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, sr_ratio=1,
                 attn_mode="direct_xor", spike_mode="lif", dvs=False, layer=0):
        super().__init__()
        self.attn = DSSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                         spike_mode=spike_mode, dvs=dvs, layer=layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
                               spike_mode=spike_mode, layer=layer)

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook