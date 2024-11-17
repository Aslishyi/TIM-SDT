import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.in_lif = MyNode(tau=2.0, v_threshold=0.3, layer_by_layer=False, step=1)
        self.out_lif = MyNode(tau=2.0, v_threshold=0.5, layer_by_layer=False, step=1)

        # Define tim_alpha as a trainable parameter
        # Correct initialization of tim_alpha as a trainable parameter
        self.tim_alpha = nn.Parameter(torch.tensor([TIM_alpha], dtype=torch.float32) + torch.rand(1) * 0.01,
                                      requires_grad=True)

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

        output = []
        x_tim = torch.empty_like(x[0])

        for i in range(T):
            if i == 0:
                x_tim = x[i]
                output.append(x_tim)
            else:
                x_tim = self.interactor(x_tim.flatten(0, 1)).reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                # Make sure to preserve the computation graph
                x_tim = x_tim + 0  # A dummy operation to ensure `x_tim` stays connected to the graph
                output.append(x_tim)

        output = torch.stack(output)
        return output

