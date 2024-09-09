"""
Extend SEAttention by implementing a multi-resolution attention mechanism
Create two versions of the input feature map: the original and a single downsampled version
Apply the SEAttention block to each version, and then upsample the downsampled attention-weighted feature map back to the original resolution
Combine these maps to form a final attention map
Modify the forward function to include these steps while optimizing for computational efficiency
Evaluate the effectiveness by comparing detection performance on small targets using synthetic datasets, assessing both qualitative and quantitative improvements over the baseline SEAttention

"""

# Modified code

import numpy as np
import torch
from torch import nn
from torch.nn import init
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

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Original SEAttention on the original feature map
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        out1 = x * y1.expand_as(x)
        
        # Downsample the feature map using interpolation
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # SEAttention on the downsampled feature map
        y2 = self.avg_pool(x_down).view(b, c)
        y2 = self.fc(y2).view(b, c, 1, 1)
        out2 = x_down * y2.expand_as(x_down)
        
        # Upsample back to the original resolution
        out2_upsampled = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)
        
        # Combine attention maps
        out_combined = out1 + out2_upsampled
        
        return out_combined

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)

# I am done