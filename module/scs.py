import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from timm.models.layers import to_2tuple

class Optimized_SCS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W

        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dims // 8)
        self.lif1 = self._get_spiking_layer(spike_mode)

        # 第二层卷积
        self.conv2 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dims // 4)
        self.lif2 = self._get_spiking_layer(spike_mode)

        # 第三层卷积
        self.conv3 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dims // 2)
        self.lif3 = self._get_spiking_layer(spike_mode)

        # 第四层卷积
        self.conv4 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dims)
        self.lif4 = self._get_spiking_layer(spike_mode)

        # 跳跃连接
        self.shortcut_conv = nn.Conv2d(embed_dims // 8, embed_dims, kernel_size=1, stride=1, bias=False)

        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _get_spiking_layer(self, spike_mode):
        if spike_mode == "lif":
            return MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            return MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        else:
            raise ValueError("Unsupported spike_mode")

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1

        # 第一层
        x = self.conv1(x.flatten(0, 1))
        x = self.bn1(x).reshape(T, B, -1, H // ratio, W // ratio)
        x = self.lif1(x)
        shortcut = self.shortcut_conv(x.flatten(0, 1)).reshape(T, B, -1, H // ratio, W // ratio)

        if self.pooling_stat[0] == "1":
            x = self.pool1(x.flatten(0, 1)).reshape(T, B, -1, H // 2, W // 2)
            ratio *= 2

        # 第二层
        x = self.conv2(x.flatten(0, 1))
        x = self.bn2(x).reshape(T, B, -1, H // ratio, W // ratio)
        x = self.lif2(x)

        if self.pooling_stat[1] == "1":
            x = self.pool2(x.flatten(0, 1)).reshape(T, B, -1, H // 4, W // 4)
            ratio *= 2

        # 第三层
        x = self.conv3(x.flatten(0, 1))
        x = self.bn3(x).reshape(T, B, -1, H // ratio, W // ratio)
        x = self.lif3(x)

        if self.pooling_stat[2] == "1":
            x = self.pool3(x.flatten(0, 1)).reshape(T, B, -1, H // 8, W // 8)
            ratio *= 2

        # 第四层（结合跳跃连接）
        x = self.conv4(x.flatten(0, 1))
        x = self.bn4(x).reshape(T, B, -1, H // ratio, W // ratio)
        x = self.lif4(x)
        x = x + shortcut  # 跳跃连接

        return x, (H // ratio, W // ratio), hook
