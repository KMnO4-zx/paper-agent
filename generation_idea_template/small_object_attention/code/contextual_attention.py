"""
Extend SEAttention by adding a global context block that pools the entire feature map into a context vector
Use this vector to inform a spatial attention recalibration mechanism, which is applied after channel attention
Implement this by adding a global context pooling layer and a recalibration module in the forward function
Evaluate the model's effectiveness by comparing feature maps and performance metrics on synthetic datasets designed to mimic scenarios with small target detection challenges

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class GlobalContextBlock(nn.Module):
    def __init__(self, channel):
        super(GlobalContextBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.global_context = GlobalContextBlock(channel)

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
        # Channel attention
        channel_attention = self.avg_pool(x).view(b, c)
        channel_attention = self.channel_fc(channel_attention).view(b, c, 1, 1)
        x = x * channel_attention.expand_as(x)

        # Spatial attention using global context
        spatial_attention = self.global_context(x)
        x = x * spatial_attention

        return x
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)