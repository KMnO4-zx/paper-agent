"""
Extend SEAttention by adding a learnable gating mechanism that dynamically adjusts attention weights based on input complexity
Implement this by introducing a gating layer that takes input feature statistics (e
g
, variance, mean) to modulate the balance between the original feature and the recalibrated attention feature
Modify the forward function to integrate this gating mechanism after the channel attention
Evaluate the model's performance by comparing it with the baseline SEAttention and other modifications, using synthetic datasets for small target detection and analyzing adaptive behavior through feature map visualizations

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
        # Gating mechanism
        self.gate_fc = nn.Sequential(
            nn.Linear(2, channel // reduction, bias=False),  # Assuming input feature statistics
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 1, bias=False),
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
        attention_weights = self.fc(y).view(b, c, 1, 1)
        
        # Compute input feature statistics
        mean = x.mean(dim=[2, 3], keepdim=False).view(b, c, 1, 1)
        variance = x.var(dim=[2, 3], keepdim=False).view(b, c, 1, 1)
        feature_stats = torch.cat((mean, variance), dim=1).view(b, 2, 1, 1)
        
        # Gating mechanism
        gate_value = self.gate_fc(feature_stats.view(b, 2)).view(b, 1, 1, 1)
        
        # Modulate attention feature with gating mechanism
        modulated_feature = gate_value * x + (1 - gate_value) * (x * attention_weights.expand_as(x))
        
        return modulated_feature
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)