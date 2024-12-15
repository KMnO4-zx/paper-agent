"""
Implement a hierarchical structure within the CoordAtt module, using two sequential layers
The first layer applies channel-wise attention with 1x1 convolutions, focusing on channel dependencies
The second layer implements spatial attention with adaptive pooling and convolutions to capture spatial dependencies
Adjust the forward method to process these layers in sequence
Evaluate the enhanced attention mechanism by testing on a small benchmark dataset, comparing improvements in feature representation and performance to the original CoordAtt while monitoring computational efficiency

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

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # First layer: channel-wise attention
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act1 = h_swish()
        
        # Second layer: spatial attention
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(mip)
        self.act2 = h_swish()

        self.conv_out = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x

        # Channel-wise attention
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        # Spatial attention
        n, c, h, w = y.size()
        x_h = self.pool_h(y)
        x_w = self.pool_w(y).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.bn2(y)
        y = self.act2(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        out = self.conv_out(out)
        
        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)