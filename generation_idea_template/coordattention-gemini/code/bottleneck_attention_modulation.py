"""
Modify the `CoordAtt` module
After pooling height and width features, concatenate them along the channel dimension
Apply a single 1x1 convolution to the concatenated feature map
Then, apply a bottleneck layer consisting of a 1x1 convolution, a non-linearity (e
g
, ReLU), and another 1x1 convolution, followed by a sigmoid activation to produce an attention map
Use this attention map to modulate the *original* pooled height and width features *separately* before concatenating them
Finally, feed the modulated concatenated map to the shared `conv1`
In the `__init__` function, add three 1x1 convolution layers, one for the initial concatenated feature transformation and two for the bottleneck attention map generation, and a non-linearity
In the `forward` function, implement the concatenation, the initial transformation, the bottleneck attention map generation, the separate modulation of the pooled height and width features, and finally the concatenation before feeding to the shared `conv1`
The rest of the forward pass remains the same
Compare output with the baseline using same test input, observe changes

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

        # Bottleneck attention layers
        self.conv_concat = nn.Conv2d(inp * 2, mip, kernel_size=1, stride=1, padding=0) # For initial transformation
        self.bottleneck_conv1 = nn.Conv2d(mip, mip // 2, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.bottleneck_conv2 = nn.Conv2d(mip // 2, mip, kernel_size=1, stride=1, padding=0)
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate pooled features
        y = torch.cat([x_h, x_w], dim=2)

        # Initial transformation
        y_concat = self.conv_concat(y)
        
        # Bottleneck attention map
        attn = self.bottleneck_conv1(y_concat)
        attn = self.relu(attn)
        attn = self.bottleneck_conv2(attn).sigmoid()
        
        # Apply bottleneck attention
        y_attn = y_concat * attn
        
        # Shared conv layer
        y = self.conv1(y_attn)
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