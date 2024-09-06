"""
Implement a dynamic attention mechanism that selects between spatial and channel attentions based on input characteristics
Develop a decision layer that analyzes input features and outputs a preference score for each attention type
Modify the SEAttention class to incorporate spatial attention
Use the decision layer to dynamically apply spatial or channel attention
Evaluate the model's performance on small target detection tasks by analyzing precision, recall, and F1-score, comparing against the baseline SEAttention model and other enhanced models

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class DynamicAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # Channel Attention Components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_channel = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention Components
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # Decision Layer
        self.decision_layer = nn.Sequential(
            nn.Linear(channel, 2),
            nn.Softmax(dim=1)
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

        # Channel Attention
        y_channel = self.avg_pool(x).view(b, c)
        y_channel = self.fc_channel(y_channel).view(b, c, 1, 1)

        # Spatial Attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y_spatial = torch.cat([max_pool, avg_pool], dim=1)
        y_spatial = self.conv_spatial(y_spatial)

        # Decision Layer
        x_flat = x.view(b, c, -1).mean(dim=2)  # Global feature descriptor
        decision_scores = self.decision_layer(x_flat)  # [prob_channel, prob_spatial]

        # Weighted combination
        out = x * (decision_scores[:, 0].view(b, 1, 1, 1) * y_channel.expand_as(x) + 
                   decision_scores[:, 1].view(b, 1, 1, 1) * y_spatial.expand_as(x))

        return out

if __name__ == '__main__':
    model = DynamicAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)