"""
Extend SEAttention by adding a lightweight geometric transformation layer that applies controlled transformations (e.g, small rotations, translations) to the input feature map
Integrate a transformation-aware attention mechanism that recalibrates feature maps based on invariant patterns across these transformations
Modify the forward function to include these geometric transformations and subsequent attention recalibration
Evaluate the model's effectiveness by comparing detection accuracy and visual focus of attention maps on synthetic datasets, particularly observing improvements in small target detection

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
import torchvision.transforms as T

class GeometricTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = T.Compose([
            T.RandomAffine(degrees=5, translate=(0.05, 0.05))
        ])

    def forward(self, x):
        # Apply geometric transformation
        return self.transforms(x)

class TransformationAwareAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.weight = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x, transformed_x):
        # Recalibrate feature maps based on invariant patterns
        attention_map = torch.sigmoid(self.weight)
        return x * attention_map + transformed_x * (1 - attention_map)

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
        self.geo_transform = GeometricTransformLayer()
        self.trans_attention = TransformationAwareAttention(channel)

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
        # Apply geometric transformation
        transformed_x = self.geo_transform(x)
        # SE attention
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        se_attention = x * y.expand_as(x)
        # Transformation-aware attention
        attention_output = self.trans_attention(se_attention, transformed_x)
        return attention_output
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)