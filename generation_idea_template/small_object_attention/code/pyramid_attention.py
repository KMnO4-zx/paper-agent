"""
Extend SEAttention by incorporating a pyramid pooling layer to generate multi-scale context features
Implement this by adding a pyramid pooling module that extracts pooled features at different scales
Apply a unified attention mechanism across these pooled features to recalibrate the feature map
Modify the forward function to include pyramid pooling and attention application
Evaluate the model's effectiveness by comparing detection performance on small and distributed targets, using visualization techniques and quantitative analysis on synthetic datasets

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pool_sizes])

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pyramids = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        return torch.cat(pyramids, dim=1)

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.pyramid_pooling = PyramidPooling(channel, pool_sizes)
        self.attention_conv = nn.Conv2d(channel * len(pool_sizes), channel, kernel_size=1, bias=False)
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

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.pyramid_pooling(x)
        x = self.attention_conv(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)