"""
Modify the `CoordAtt` module
After pooling height and width features, apply a *single 1x1 convolution* to each of the pooled feature maps *separately*
This 1x1 conv will output *two channels*
The first channel will represent the transformed feature, and the second channel will represent the *dynamic weight*
Apply a *sigmoid activation* to the dynamic weight channel
Then, perform a weighted addition of the transformed height and width features using their respective sigmoid-activated dynamic weights
Feed the result into the shared `conv1`
In the `__init__` function, add *two 1x1 convolution layers*, one for each of height and width, each of which output *two channels*
In the `forward` function, implement the separate convolutions, the separation of the two output channels into transformed feature and dynamic weight, the application of the sigmoid activation to the dynamic weight channel, the weighted addition using these sigmoid-activated dynamic weights, before passing the result to the shared `conv1`
The rest of the forward pass remains the same
Compare output with the baseline using the same test input and observe the changes
This involves modifying `__init__` to include the 1x1 conv layers with two output channels, and `forward` to implement the channel separation, sigmoid activation and dynamic fusion

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
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # Add two 1x1 conv layers, one for h and one for w, each outputting 2 channels
        self.conv_h_sep = nn.Conv2d(inp, 2, kernel_size=1, stride=1, padding=0)
        self.conv_w_sep = nn.Conv2d(inp, 2, kernel_size=1, stride=1, padding=0)
        self.conv_h_expand = nn.Conv2d(1, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w_expand = nn.Conv2d(1, inp, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Apply separate 1x1 convs to pooled features
        x_h_sep = self.conv_h_sep(x_h)
        x_w_sep = self.conv_w_sep(x_w)

        # Separate transformed feature and dynamic weight
        x_h_trans, x_h_weight = torch.split(x_h_sep, [1, 1], dim=1)
        x_w_trans, x_w_weight = torch.split(x_w_sep, [1, 1], dim=1)


        # Apply sigmoid to dynamic weights
        a_h = x_h_weight.sigmoid()
        a_w = x_w_weight.sigmoid()

        # Expand channels before weighted addition
        x_h_trans = self.conv_h_expand(x_h_trans)
        x_w_trans = self.conv_w_expand(x_w_trans)

        # Weighted addition of transformed features
        y =  x_h_trans * a_h + x_w_trans.permute(0, 1, 3, 2) * a_w


        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)


        out = identity * y

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)