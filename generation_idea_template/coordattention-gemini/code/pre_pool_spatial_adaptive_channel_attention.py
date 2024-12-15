"""
Modify the `CoordAtt` module
Introduce a spatial adaptive channel attention module before the pooling operations
This module will consist of global average pooling and global max pooling
A 1x1 convolution will be used to generate a spatial weight map for the max pooling output
The weighted max pooling output is then added element-wise to the average pooling output
This result is then passed through a 1x1 convolution, a ReLU activation, and a sigmoid activation
The output of the channel attention will be used to modulate the input feature map before the height and width pooling
Modify the `__init__` function to include the channel attention module and the 1x1 convolution for weight map generation
Modify the `forward` function to implement the channel attention, the weighted sum of the pooling outputs using the spatial weight map, modulation of the input feature map, and then the rest of the operations
Compare the output with the baseline using the same test input and observe the changes
This involves adding global average pooling, global max pooling, a 1x1 conv for spatial weight map, a 1x1 conv, ReLU and sigmoid, and modifying the forward pass to apply the attention before pooling

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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mip = max(8, inp // reduction)
        self.conv_reduce = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        
        self.spatial_weight = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_reduced = self.conv_reduce(x)
        avg_out = self.avg_pool(x_reduced)
        max_out = self.max_pool(x_reduced)
        
        spatial_weight = self.spatial_weight(max_out).sigmoid()
        
        channel_att = avg_out + max_out * spatial_weight
        
        channel_att = self.bn1(channel_att)
        channel_att = self.act(channel_att)
        
        x = x * channel_att

        x_h = nn.AdaptiveAvgPool2d((None, 1))(x)
        x_w = nn.AdaptiveAvgPool2d((1, None))(x).permute(0, 1, 3, 2)


        y = torch.cat([x_h, x_w], dim=2)
        
        y = self.bn1(self.conv_reduce(y))
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