"""
Introduce a Channel Mixing Block (CMB) within the CoordAtt module
Implement grouped convolutions in the CMB to capture channel-wise dependencies efficiently
Modify the CoordAtt class to include a new CMB after the initial convolutional layers
Evaluate the impact of channel mixing on feature representation by testing on a small benchmark dataset, comparing the performance and computational efficiency against the original CoordAtt and other variants

"""

# Modified code
import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ChannelMixingBlock(nn.Module):
    def __init__(self, channels, groups=4):
        super(ChannelMixingBlock, self).__init__()
        self.groups = groups
        self.grouped_conv = nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = h_swish()

    def forward(self, x):
        x = self.grouped_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32, groups=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # Introduce Channel Mixing Block
        self.cmb = ChannelMixingBlock(mip, groups=groups)

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Pass through Channel Mixing Block
        y = self.cmb(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)