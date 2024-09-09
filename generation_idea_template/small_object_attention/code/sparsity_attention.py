"""
Extend SEAttention by incorporating a sparsity-promoting transformation within the attention mechanism
Implement a sparse encoding step using a learned thresholding layer, applied to the input feature maps before the existing attention recalibration
This thresholding layer will dynamically adjust based on the input characteristics to promote sparsity efficiently
Modify the forward function to include this sparsity transformation and evaluate its impact by comparing detection performance on synthetic datasets with baseline SEAttention
Use visualization of attention maps to assess enhanced focus on critical features and improved noise suppression

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
        self.sparsity_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.attention_layer = nn.Sequential(
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

    def apply_sparsity(self, x, threshold):
        """Apply sparsity mask based on the dynamic threshold."""
        return x * (x > threshold).float()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Sparsity-promoting transformation
        sparsity_threshold = self.avg_pool(x).view(b, c)
        sparsity_threshold = self.sparsity_layer(sparsity_threshold).view(b, c, 1, 1)
        x = self.apply_sparsity(x, sparsity_threshold)
        
        # SEAttention mechanism
        y = self.avg_pool(x).view(b, c)
        y = self.attention_layer(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)