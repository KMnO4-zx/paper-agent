"""
Extend SEAttention by integrating a semantic attention module that utilizes a simple pooling strategy to identify salient semantic features
Implement this by adding a global max pooling layer to extract prominent features, followed by a learnable attention layer that assigns weights based on semantic relevance
Modify the forward function to incorporate this semantic attention after the channel attention
Evaluate the model's effectiveness by comparing detection performance on synthetic datasets, focusing on improvements in semantic understanding and detection accuracy

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
        # Semantic attention components
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.semantic_fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
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
        
        # Channel attention
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)

        # Semantic attention
        z = self.global_max_pool(x).view(b, c)
        z = self.semantic_fc(z).view(b, c, 1, 1)
        x = x * z.expand_as(x)

        return x

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)