import torch
from roboninja.model.nn_utils import Conv, Down, MLP
from torch import nn


class CloseLoopPolicyModel(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(CloseLoopPolicyModel, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.inc = Conv(self.in_channels, 16)   # 256
        self.down1 = Down(16, 32)               # 128
        self.down2 = Down(32, 64)               # 64
        self.down3 = Down(64, 128)              # 32
        self.down4 = Down(128, 256)             # 16
        self.down5 = Down(256, 256)             # 8
        self.down6 = Down(256, 256)             # 4
        self.down7 = nn.Conv2d(256, 256, 4)     # 1
        self.mlp = MLP([256, 256, 256, out_dim], last_relu=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = torch.relu(torch.flatten(x8, start_dim=1))
        output = self.mlp(x9)
        return output
