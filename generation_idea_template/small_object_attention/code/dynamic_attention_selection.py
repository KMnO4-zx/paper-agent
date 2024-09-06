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

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        
        # Channel Attention Components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention Components
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(channel // reduction, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

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

    def channel_attention(self, x, b, c):
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def spatial_attention(self, x, b, c, h, w):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Decision layer based on input features
        avg_features = self.avg_pool(x).view(b, c)
        decision = self.decision_layer(avg_features)
        
        # Split decision into channel and spatial attention weights
        channel_weight, spatial_weight = decision[:, 0], decision[:, 1]
        
        # Apply attention based on decision weights
        channel_attended = self.channel_attention(x, b, c) * channel_weight.view(b, 1, 1, 1)
        spatial_attended = self.spatial_attention(x, b, c, h, w) * spatial_weight.view(b, 1, 1, 1)
        
        return channel_attended + spatial_attended
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)