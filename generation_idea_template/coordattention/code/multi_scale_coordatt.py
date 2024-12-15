"""
Add parallel convolutional branches with kernel sizes 1x1, 3x3, and 5x5 to `CoordAtt` to capture multi-scale features
Each branch should have its own convolutional layer followed by batch normalization and activation
Concatenate the outputs of these branches before combining with the original coordinate attention features
Modify the `forward` method to integrate these multi-scale features before applying attention weights
Evaluate the enhancement in feature representation by testing on a small benchmark dataset and comparing the modified module's performance to the original, while also monitoring computational overhead

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
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # Original coordinate attention components
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Multi-scale feature branches
        self.conv_1x1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn_1x1 = nn.BatchNorm2d(mip)

        self.conv_3x3 = nn.Conv2d(inp, mip, kernel_size=3, stride=1, padding=1)
        self.bn_3x3 = nn.BatchNorm2d(mip)

        self.conv_5x5 = nn.Conv2d(inp, mip, kernel_size=5, stride=1, padding=2)
        self.bn_5x5 = nn.BatchNorm2d(mip)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Coordinate attention path
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Multi-scale feature branches
        y_1x1 = self.bn_1x1(self.conv_1x1(x))
        y_3x3 = self.bn_3x3(self.conv_3x3(x))
        y_5x5 = self.bn_5x5(self.conv_5x5(x))

        multi_scale_features = torch.cat([y_1x1, y_3x3, y_5x5], dim=1)
        multi_scale_features = self.act(multi_scale_features)

        # Combine multi-scale features with coordinate attention
        out = identity * a_w * a_h + multi_scale_features

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)