"""
Modify the `CoordAtt` module
Apply a 1x1 convolution to the input feature map before applying max and average pooling
Concatenate the results of the max and average pooling along the channel dimension
Apply a lightweight spatial attention module to the concatenated map which will have a 3x3 depthwise convolution, followed by a 1x1 convolution and finally a sigmoid activation
This spatial attention module modulates the combined attention map before it is split into a_h and a_w
Modify the `__init__` to include the 1x1 convolution before the pooling layers, and the lightweight spatial attention module
Modify the `forward` to implement the new pooling and modulation scheme
The output can be compared to the baseline using the same test input and observing the changes in output
This involves modifying `__init__` to incorporate pre-pooling conv, depthwise conv, 1x1 conv and sigmoid, and `forward` to implement the pooling and spatial attention

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
        self.pre_conv = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0) # 1x1 conv before pooling
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Spatial Attention Module
        self.spatial_conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, groups=1) # Depthwise conv
        self.spatial_sigmoid = nn.Sigmoid()


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x = self.pre_conv(x) # Apply 1x1 convolution

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        #Spatial Attention
        attention_map = torch.cat([x_h, x_w.permute(0,1,3,2)], dim=1)
        attention_map = self.spatial_conv1(attention_map)
        attention_map = self.spatial_sigmoid(attention_map)
        attention_map = F.interpolate(attention_map, size=(h, w), mode='bilinear', align_corners=False)
        
        y = torch.cat([x_h, x_w], dim=2)
        
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)

        a_h = F.interpolate(a_h, size=(h, w), mode='bilinear', align_corners=False).sigmoid()
        a_w = F.interpolate(a_w, size=(h, w), mode='bilinear', align_corners=False).sigmoid()


        out = identity * a_w * a_h * attention_map

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)