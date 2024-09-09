"""
Extend SEAttention by adding a spatial attention layer
Implement this by introducing a convolutional layer that outputs a spatial attention map with the same height and width as the input feature map
In the forward function, apply spatial attention by element-wise multiplying the spatial attention map with the input feature map, followed by the existing channel attention
Evaluate the model's effectiveness by comparing the output feature maps against those from the original SEAttention, using input tensors of varying scales and complexities
Performance can be assessed by visual inspection of feature maps and quantitative analysis using synthetic datasets if available

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
        
        # Spatial attention layer
        self.spatial_conv = nn.Conv2d(channel, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

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
        # Spatial attention
        spatial_att = self.spatial_conv(x)
        spatial_att = self.spatial_sigmoid(spatial_att)
        x = x * spatial_att

        # Channel attention
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)