"""
Implement an internal attention bootstrapping mechanism where SEAttention periodically saves and analyzes its attention distribution at various training stages
Modify the training routine to adjust current attention maps to better align with or improve upon these previously saved distributions, focusing on enhancing small target detection capabilities
Evaluate attention map alignment and detection performance improvements over baseline SEAttention using synthetic datasets

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

    def __init__(self, channel=512, reduction=16, save_interval=10):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.attention_history = []
        self.save_interval = save_interval
        self.training_step = 0

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

        # Save attention distribution periodically
        if self.training and self.training_step % self.save_interval == 0:
            self.attention_history.append(y.detach().clone())
        
        # Adjust attention maps based on saved distributions
        if self.attention_history:
            historical_attention = self.attention_history[-1]
            y = self._adjust_attention(y, historical_attention)
        
        self.training_step += 1
        return x * y.expand_as(x)

    def _adjust_attention(self, current_attention, historical_attention):
        # Simple example of adjustment: interpolate between current and historical attention
        adjusted_attention = (current_attention + historical_attention) / 2
        return adjusted_attention

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)