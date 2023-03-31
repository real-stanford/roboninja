
from roboninja.model.nn_utils import Conv, Down, Up
from torch import nn


class StateEstimationModel(nn.Module):
    def __init__(self, in_channels):
        super(StateEstimationModel, self).__init__()
        self.n_channels = in_channels

        self.inc = Conv(in_channels, 16)             # 256
        self.down1 = Down(16, 32)                   # 128
        self.down2 = Down(32, 64)                   # 64
        self.down3 = Down(64, 128)                  # 32
        self.down4 = Down(128, 256)                 # 16
        self.down5 = Down(256, 256)                 # 8

        self.up1 = Up(512, 128, bilinear=True)
        self.up2 = Up(256, 64, bilinear=True)
        self.up3 = Up(128, 32, bilinear=True)
        self.up4 = Up(64, 16, bilinear=True)
        self.up5 = Up(32, 16, bilinear=True)
        self.outc = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
