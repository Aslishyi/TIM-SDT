import torch
import torch.nn as nn
from braincog.model_zoo.base_module import BaseModule
from utils.MyNode import MyNode

class TIM(BaseModule):
    def __init__(self, dim=256, encode_type='direct', in_channels=None, TIM_alpha=0.5):
        super().__init__(step=1, encode_type=encode_type)

        if in_channels is None:
            in_channels = dim

        self.interactor = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        # 降低 LIF 阈值，确保激活
        self.in_lif = MyNode(tau=2.0, v_threshold=0.1, layer_by_layer=False, step=1)
        self.out_lif = MyNode(tau=2.0, v_threshold=0.2, layer_by_layer=False, step=1)

        self.tim_alpha = TIM_alpha

        # 初始化 interactor 权重
        nn.init.kaiming_normal_(self.interactor.weight, mode='fan_out', nonlinearity='relu')
        if self.interactor.bias is not None:
            nn.init.constant_(self.interactor.bias, 0)

    def forward(self, x):
        self.reset()
        if len(x.shape) == 4:  # Input without temporal dimension
            B, H, N, CoH = x.shape
            T = 1
            x = x.unsqueeze(0)
        elif len(x.shape) == 5:  # Correct shape
            T, B, H, N, CoH = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Adjust the interactor to match the correct input channels
        if self.interactor.in_channels != CoH:
            self.interactor = nn.Conv1d(
                in_channels=CoH,
                out_channels=CoH,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ).to(x.device)
            nn.init.kaiming_normal_(self.interactor.weight, mode='fan_out', nonlinearity='relu')
            if self.interactor.bias is not None:
                nn.init.constant_(self.interactor.bias, 0)

        output = []
        x_tim = torch.empty_like(x[0])

        for i in range(T):
            if i == 0:
                x_tim = x[i]
                output.append(x_tim)
            else:
                interactor_out = self.interactor(x_tim.flatten(0, 1))
                x_tim = interactor_out.reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                x_tim = x_tim + 0  # Preserve computation graph
                output.append(x_tim)

        output = torch.stack(output)
        return output