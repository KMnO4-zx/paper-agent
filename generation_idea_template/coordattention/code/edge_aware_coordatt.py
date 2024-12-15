"""
Integrate a lightweight edge detection layer, such as a Sobel filter or small trainable convolutional layer, within the CoordAtt module
Modify the forward method to compute edge maps and integrate them by modulating the attention weights
Evaluate the impact on feature representation by testing on a small benchmark dataset, comparing performance to the original CoordAtt, and monitoring computational efficiency

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Edge detection using Sobel filters
        self.sobel_x = nn.Conv2d(inp, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(inp, 1, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(1, inp, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(1, inp, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        identity = x

        # Compute edge maps
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_map = edge_map.sigmoid()  # Normalize edge map

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Integrate edge map by modulating attention
        out = identity * a_w * a_h * edge_map

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)