"""
Enhance the `CoordAtt` module by integrating a dynamic complexity adjustment mechanism
Use a simple heuristic based on feature map variance to determine complexity
Route the input through either a lightweight or complex processing path: a basic path for low variance features and an enhanced path for high variance features
Modify the forward method to include this decision mechanism and dynamically adjust processing
Evaluate the module's adaptability on a small benchmark dataset, assessing improvements in feature discrimination and computational efficiency over the original and other variants

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

        # Lightweight path
        self.light_conv = nn.Sequential(
            nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mip),
            h_swish()
        )

        # Complex path
        self.complex_conv = nn.Sequential(
            nn.Conv2d(inp, mip, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mip),
            h_swish(),
            nn.Conv2d(mip, mip, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mip),
            h_swish()
        )

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        # Calculate feature map variance
        variance = x.var(dim=(2, 3), keepdim=True).mean()

        # Choose path based on variance
        if variance < 0.5:  # Threshold can be tuned
            y = self.light_conv(x)
        else:
            y = self.complex_conv(x)

        n, c, h, w = y.size()
        x_h = self.pool_h(y)
        x_w = self.pool_w(y).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.light_conv(y)

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