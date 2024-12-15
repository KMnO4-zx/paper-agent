"""
Modify the CoordAtt module to introduce a probabilistic attention mechanism
Implement a stochastic gating mechanism where attention weights are sampled from a Gaussian distribution with learnable parameters (mean and variance)
Adjust the forward method to compute these probabilistic attention weights and integrate them into the feature modulation process
Evaluate the impact on feature representation and robustness by testing on a small benchmark dataset, comparing the performance to the original CoordAtt and other variants, and analyzing the uncertainty estimates

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

class ProbabilisticAttention(nn.Module):
    def __init__(self, channels):
        super(ProbabilisticAttention, self).__init__()
        self.mean = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_var = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        std = torch.exp(0.5 * self.log_var)
        epsilon = torch.randn_like(std)
        attention_weights = self.mean + std * epsilon
        return torch.sigmoid(attention_weights)

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

        # Introduce probabilistic attention
        self.prob_att_h = ProbabilisticAttention(inp)
        self.prob_att_w = ProbabilisticAttention(inp)

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

        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)

        # Apply probabilistic attention
        pa_h = self.prob_att_h(a_h)
        pa_w = self.prob_att_w(a_w)

        out = identity * pa_w * pa_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)