"""
Implement a simple heuristic-based mechanism within the CoordAtt module to dynamically adjust attention parameters based on input feature map statistics such as variance or entropy
Modify the forward method to incorporate this mechanism, allowing it to adaptively configure the attention strategy
Evaluate the adaptability and performance on a small benchmark dataset, comparing it with the original CoordAtt and other versions, focusing on accuracy, feature representation quality, and computational efficiency
The heuristic could be a rule-based system or a lightweight decision tree

在CoordAtt模块中实现一个基于输入特征图统计（如方差或熵）的简单启发式机制，以动态调整注意力参数。
修改forward方法以包含此机制，使其能够自适应地配置注意力策略。
通过在一个小型基准数据集上评估其适应性和性能，
比较与原始CoordAtt和其他版本的差异，重点关注准确性、特征表示质量和计算效率。启发式方法可以是基于规则的系统或轻量级决策树。
"""

# xaa 可以试试 xxx实测不行 学不到任何能力

# Modified code

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class ADACoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(ADACoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def compute_variance(self, x):
        # Calculate the variance of the feature map
        return torch.var(x, dim=(2, 3), keepdim=True)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        
        # Compute variance and use it to adjust the attention
        variance = self.compute_variance(x)
        
        # Heuristic rule: if variance is high, reduce the impact of attention
        # Scale factor ranges from 0.5 to 1.0 based on variance
        scale_factor = torch.clamp(1.0 - 0.5 * (variance / variance.max()), min=0.5, max=1.0)

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

        # Apply the scale factor to the attention maps
        a_h = a_h * scale_factor
        a_w = a_w * scale_factor

        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = ADACoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)