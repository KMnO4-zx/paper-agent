"""
Extend SEAttention by incorporating a meta-attention mechanism
Implement this by introducing a secondary attention module, such as an additional SE block or a simple linear transformation, to refine the channel weights produced by the original SEAttention
Modify the forward function to apply this meta-attention after the original attention recalibration while maintaining computational efficiency
Evaluate the model's performance by comparing feature maps and conducting quantitative assessments on synthetic datasets, focusing on improvements in attention focus and detection performance on small targets

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

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Meta-attention module: an additional SE block to refine channel weights
        self.meta_fc = nn.Sequential(
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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)  # Original SE block
        # Apply meta-attention to refine channel weights
        y_meta = self.meta_fc(y).view(b, c, 1, 1)
        return x * y_meta.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)