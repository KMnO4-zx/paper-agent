"""
Develop a meta-attention module that infers a task descriptor from input data characteristics
Integrate this module within the SEAttention framework to modulate its parameters dynamically
Implement functions to extract task descriptors and modify SEAttention weights based on these descriptors
Evaluate the model's adaptability and performance across diverse small target detection tasks using precision, recall, and F1-score, comparing its performance against the baseline SEAttention model and other enhanced models

"""

# Improved Code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class TaskAdaptiveAttention(nn.Module):
    """A module to infer task descriptors from input data characteristics."""
    
    def __init__(self, channel=512):
        super(TaskAdaptiveAttention, self).__init__()
        self.task_descriptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Extract task descriptor from input data."""
        return self.task_descriptor(x)
        

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.task_adaptive_attention = TaskAdaptiveAttention(channel)

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
        # Task descriptor influences SEAttention weights
        task_descriptor = self.task_adaptive_attention(x)
        pooled = self.avg_pool(x).view(b, c)
        se_weights = self.fc(pooled).view(b, c, 1, 1)
        adaptive_weights = task_descriptor.view(b, c, 1, 1)
        return x * se_weights.expand_as(x) * adaptive_weights.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)