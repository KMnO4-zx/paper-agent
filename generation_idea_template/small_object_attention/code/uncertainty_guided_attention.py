"""
Integrate an uncertainty estimation module within the SEAttention framework
Develop functions to compute uncertainty scores for different regions in the input feature map, using methods such as Monte Carlo Dropout or entropy-based measures
Modify the SEAttention class to incorporate these uncertainty scores into the attention mechanism, adjusting attention weights based on uncertainty
Evaluate the model's performance on small target detection tasks using metrics such as precision, recall, and F1-score, while also analyzing the uncertainty estimation's impact on detection accuracy
Compare results with the baseline SEAttention model and other enhanced models

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

    def __init__(self, channel=512, reduction=16, dropout_rate=0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.uncertainty_module = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0),
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
        
        # Estimate uncertainty
        uncertainty_scores = self.uncertainty_module(x)
        
        # Apply dropout for Monte Carlo estimation
        x_dropped = self.dropout(x)
        
        # Original SEAttention mechanism
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Adjust attention weights based on uncertainty
        adjusted_attention = y * uncertainty_scores
        
        return x_dropped * adjusted_attention.expand_as(x_dropped)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)