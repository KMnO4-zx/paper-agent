"""
Expand SEAttention to include a spatial attention mechanism
Implement spatial attention by adding two convolutional layers, followed by a softmax activation to produce a spatial attention map
Combine the spatial attention map with the SE channel attention output through element-wise multiplication
Evaluate performance improvements using metrics such as precision, recall, and F1-score on small target detection tasks
Compare these results to the baseline SEAttention model's performance

"""

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F


class SEAttentionWithSpatial(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_conv1 = nn.Conv2d(channel, channel // 8, kernel_size=7, padding=3, bias=False)
        self.spatial_conv2 = nn.Conv2d(channel // 8, 1, kernel_size=7, padding=3, bias=False)
        self.softmax = nn.Softmax(dim=2)

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
        
        # Channel Attention
        y_c = self.avg_pool(x).view(b, c)
        y_c = self.fc(y_c).view(b, c, 1, 1)
        channel_attention = x * y_c.expand_as(x)
        
        # Spatial Attention
        y_s = self.spatial_conv1(channel_attention)
        y_s = F.relu(y_s)
        y_s = self.spatial_conv2(y_s)
        y_s = self.softmax(y_s.view(b, 1, h * w)).view(b, 1, h, w)
        
        # Combined Attention
        combined_attention = channel_attention * y_s
        
        return combined_attention
    
if __name__ == '__main__':
    model = SEAttentionWithSpatial()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)