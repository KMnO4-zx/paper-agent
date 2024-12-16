"""
Modify the `CoordAtt` class to incorporate a content-adaptive attention mechanism
Implement a self-attention-like operation where attention weights are computed based on the cosine similarity between features
Integrate this mechanism before the existing coordinate attention operations, allowing attention weights to be modulated based on input content
Evaluate the benefits of this approach by assessing feature representation quality and comparing performance on a small benchmark dataset against the original CoordAtt
Monitor computational efficiency and parameter count to ensure the approach remains lightweight

"""

#  创新不足
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

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Self-attention-like mechanism using cosine similarity
        self.query_conv = nn.Conv2d(inp, inp // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(inp, inp // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(inp, inp, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        identity = x

        # Compute cosine similarity based attention
        n, c, h, w = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query = query.view(n, -1, h * w)
        key = key.view(n, -1, h * w)
        value = value.view(n, -1, h * w)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention / (c ** 0.5))
        out_attention = torch.bmm(value, attention).view(n, c, h, w)

        # Existing coordinate attention operations
        x_h = self.pool_h(out_attention)
        x_w = self.pool_w(out_attention).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
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