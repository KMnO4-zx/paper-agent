"""
Modify the `CoordAtt` module
After pooling height and width features, perform an element-wise multiplication of the pooled height and width features
Concatenate the pooled height and width features
Apply a learnable parameter followed by a sigmoid activation to the element-wise multiplied feature
Perform a weighted sum of the sigmoid-activated element-wise multiplied feature and the concatenated features
Apply a group convolution instead of 1x1 convolution in the `conv1` layer
Use a small number of groups (e
g
4 or 8)
Modify the `__init__` function to include the group convolution layer and a learnable parameter
Modify the `forward` function to implement the element-wise multiplication, concatenation, sigmoid activation of the learnable parameter, weighted sum, and the group convolution before the shared `conv1` layer
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe changes
This involves element-wise multiplication, concatenation, learnable parameter with sigmoid, weighted sum, group conv, and modifying the forward pass

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


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32, groups=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        self.weight = nn.Parameter(torch.zeros(1, mip, 1, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Element-wise multiplication
        x_hw = x_h * x_w
        
        # Concatenation
        y = torch.cat([x_h, x_w], dim=2)
        
        # Sigmoid-weighted interaction
        weight = self.sigmoid(self.weight)
        x_hw = x_hw * weight
        y = y + x_hw
        
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