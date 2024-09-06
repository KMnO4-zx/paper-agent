"""
Develop a meta-attention module that infers a task descriptor from input data characteristics
Integrate this module within the SEAttention framework to modulate its parameters dynamically
Implement functions to extract task descriptors and modify SEAttention weights based on these descriptors
Evaluate the model's adaptability and performance across diverse small target detection tasks using precision, recall, and F1-score, comparing its performance against the baseline SEAttention model and other enhanced models

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class MetaAttentionModule(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.task_descriptor_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel // 8),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        task_descriptor = self.task_descriptor_extractor(x)
        return task_descriptor

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
        self.meta_attention = MetaAttentionModule(channel=channel)
        self.dynamic_fc = nn.Linear(channel // 8, channel, bias=False)

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
        task_descriptor = self.meta_attention(x)
        dynamic_weights = self.dynamic_fc(task_descriptor).unsqueeze(-1).unsqueeze(-1)
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = y * dynamic_weights.expand_as(y)
        
        return x * y.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)