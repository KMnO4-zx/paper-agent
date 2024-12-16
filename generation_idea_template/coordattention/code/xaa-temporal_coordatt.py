"""
Add a lightweight temporal attention mechanism to the CoordAtt module
Introduce 1D convolutions that operate on temporal sequences derived from input feature maps
Modify the forward method to compute temporal attention weights and integrate them with spatial attention
Evaluate the effectiveness on synthetic sequential data to assess improvements in temporal feature representations, while monitoring any additional computational cost incurred

"""

# 可以一试

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

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32, temporal_reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Temporal Attention Module
        self.temporal_conv1 = nn.Conv1d(inp, inp // temporal_reduction, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(inp // temporal_reduction)
        self.temporal_conv2 = nn.Conv1d(inp // temporal_reduction, inp, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Temporal attention computation
        temporal_x = x.view(n, c, -1)  # Reshape to (batch_size, channels, temporal_dim)
        t = self.temporal_conv1(temporal_x)
        t = self.temporal_bn1(t)
        t = F.relu(t)
        t = self.temporal_conv2(t).sigmoid()
        t = t.view(n, c, h, w)  # Reshape back to original dimensions

        out = identity * a_w * a_h * t

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)