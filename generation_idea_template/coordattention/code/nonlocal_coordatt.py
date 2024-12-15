"""
Introduce a lightweight non-local attention mechanism within the CoordAtt module
Implement a simplified version of the non-local block that captures global context effectively
Modify the forward method to first compute these non-local attention features and integrate them with the existing coordinate attention features before applying the final attention weights
Evaluate performance on a small benchmark dataset, focusing on improvements in feature representation and capturing long-range dependencies
Compare against the original CoordAtt and other variants to assess computational efficiency and accuracy improvements

"""

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


class SimplifiedNonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(SimplifiedNonLocalBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        
    def forward(self, x):
        n, c, h, w = x.size()
        
        theta = self.theta(x).view(n, c // 2, -1)
        phi = self.phi(x).view(n, c // 2, -1)
        g = self.g(x).view(n, c // 2, -1)
        
        attention = torch.bmm(theta.permute(0, 2, 1), phi)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = out.view(n, c // 2, h, w)
        out = self.out_conv(out)
        
        return x + out


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Add non-local block
        self.non_local_block = SimplifiedNonLocalBlock(inp)

    def forward(self, x):
        identity = x

        # Compute non-local features
        non_local_features = self.non_local_block(x)

        n, c, h, w = x.size()
        x_h = self.pool_h(non_local_features)
        x_w = self.pool_w(non_local_features).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

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