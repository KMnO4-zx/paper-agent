"""
Modify the SEAttention class to incorporate a dynamic scaling factor for attention weights
Implement a new function that computes scaling factors based on the spatial dimensions of feature maps
Integrate this function into the forward pass of SEAttention to adjust attention weights dynamically
Evaluate performance using precision, recall, and F1-score on small target detection tasks, comparing against the baseline SEAttention model and other enhanced models to demonstrate improvements in detecting small targets

"""

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def compute_scaling_factor(self, x):
        _, _, h, w = x.size()
        # Example scaling factor: inverse of the sum of spatial dimensions
        return 1.0 / (h + w)

    def forward(self, x):
        b, c, _, _ = x.size()
        scaling_factor = self.compute_scaling_factor(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = y * scaling_factor
        return x * y.expand_as(x)

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)