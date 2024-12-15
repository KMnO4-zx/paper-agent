"""
Modify the `CoordAtt` module
After pooling the height and width features, perform an element-wise addition of the pooled height and width features
Apply a 1x1 convolution to the summed feature map, followed by a non-linearity (e
g
, ReLU)
Then, apply *two separate* 1x1 convolutions to the fused feature map to generate projected height and width features, respectively
Feed these projected height and width features to the respective `conv_h` and `conv_w` layers
Modify the `__init__` to add the 1x1 convolution and activation for fusion, and *two additional 1x1 convolutions* for projections
Modify the `forward` to implement the element-wise addition, 1x1 convolution, activation, two projection convolutions, and feeding to the subsequent convolution layers
The rest of the forward pass remains unchanged
Compare the output with the baseline using the same test input to observe changes
This involves adding the 1x1 conv, non-linearity, two projection 1x1 convs, and modifying the `forward` pass to implement the fusion using addition and projected representations

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

        # Fusion convolution and activation
        self.fusion_conv = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)
        self.fusion_act = nn.ReLU() # Changed to ReLU

        # Projection convolutions
        self.proj_h = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)
        self.proj_w = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)


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
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Element-wise addition of pooled features
        fused_feature = x_h + x_w

        # 1x1 convolution and activation
        fused_feature = self.fusion_conv(fused_feature)
        fused_feature = self.fusion_act(fused_feature)

        # Projection convolutions
        proj_h_feature = self.proj_h(fused_feature)
        proj_w_feature = self.proj_w(fused_feature)

        a_h = self.conv_h(proj_h_feature).sigmoid()
        a_w = self.conv_w(proj_w_feature).sigmoid()
        
        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)