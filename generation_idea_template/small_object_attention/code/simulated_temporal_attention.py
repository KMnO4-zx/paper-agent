"""
Extend SEAttention by simulating temporal attention using a sliding window approach on spatial feature maps
Implement this by adding a mechanism that divides feature maps into non-overlapping subregions, treating each as a pseudo-temporal step, and applies attention across these regions using a shared attention mechanism
This should be integrated into the forward function following the channel attention
Evaluate by testing the model on datasets where small targets are embedded in varying spatial contexts within a single image, with performance assessed through quantitative metrics and visualization of feature map focus areas to compare against the original model

"""

# Modified code

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, window_size=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.window_size = window_size

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
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)

        # Simulating temporal attention via sliding window
        sw = self.window_size
        for i in range(0, h, sw):
            for j in range(0, w, sw):
                subregion = x[:, :, i:i+sw, j:j+sw]
                pooled = subregion.mean(dim=(2, 3), keepdim=True)
                x[:, :, i:i+sw, j:j+sw] = subregion * pooled

        return x

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)