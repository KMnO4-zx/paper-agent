"""
Develop a dynamic data transformation module that learns to apply optimal transformations to input data to improve small target visibility
Integrate this module into the SEAttention model by preprocessing input data before the attention mechanism
Evaluate the model's performance by comparing precision, recall, and F1-score on small target detection tasks with and without the transformation module
Analyze the effectiveness of different transformations in enhancing small target detection

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F


class DynamicDataTransformation(nn.Module):
    
    def __init__(self, transformations):
        super().__init__()
        self.transformations = transformations
        self.weights = nn.Parameter(torch.ones(len(transformations)))  # Learnable weights for each transformation
    
    def forward(self, x):
        transformed_data = [trans(x) for trans in self.transformations]
        stacked_data = torch.stack(transformed_data, dim=0)
        weights = F.softmax(self.weights, dim=0)
        weighted_sum = torch.sum(weights.view(-1, 1, 1, 1, 1) * stacked_data, dim=0)
        return weighted_sum


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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class EnhancedSEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.data_transform = DynamicDataTransformation(transformations=[
            lambda x: x,
            lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False),
            lambda x: F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        ])
        self.se_attention = SEAttention(channel, reduction)

    def forward(self, x):
        x_transformed = self.data_transform(x)
        return self.se_attention(x_transformed)
    

if __name__ == '__main__':
    model = EnhancedSEAttention()
    model.se_attention.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)