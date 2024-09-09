"""
Extend SEAttention by integrating a cross-channel attention mechanism using a multi-head attention layer
This layer computes interactions between channels to create a comprehensive attention map that enhances feature recalibration
Modify the forward function to apply this cross-channel attention before the existing channel attention
Evaluate the model by comparing outputs with those from SEAttention and other modifications, using attention map visualizations and performance on synthetic datasets designed to test inter-channel dependencies

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
    def __init__(self, channel=512, reduction=16, num_heads=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Cross-channel multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=channel, num_heads=num_heads, batch_first=True)

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
        
        # Reshape and transpose for multi-head attention
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # shape: (b, hw, c)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)
        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)  # reshape back to original input shape

        # Existing SEAttention mechanism
        y = self.avg_pool(attn_output).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return attn_output * y.expand_as(attn_output)

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)