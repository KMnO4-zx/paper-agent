"""
Extend SEAttention by implementing a color channel fusion mechanism
Add a preprocessing step that applies a shared attention mechanism to the R, G, and B channels, followed by a weighted fusion of these channels to create a comprehensive feature map
Integrate this fused feature map into the existing SEAttention architecture
Evaluate the model's effectiveness by comparing detection performance on synthetic datasets designed with varying color contrasts and subtle variations, using quantitative metrics such as precision and recall, and qualitative analysis of attention map focus

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ColorChannelFusion(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.red_attention = ChannelAttention(channel=channel, reduction=reduction)
        self.green_attention = ChannelAttention(channel=channel, reduction=reduction)
        self.blue_attention = ChannelAttention(channel=channel, reduction=reduction)
        self.weighted_fusion = nn.Conv2d(channel * 3, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # Assuming input x shape is (batch_size, 3, height, width)
        red, green, blue = torch.split(x, 1, dim=1)
        red = self.red_attention(red)
        green = self.green_attention(green)
        blue = self.blue_attention(blue)
        
        # Concatenate along channel dimension
        fused = torch.cat([red, green, blue], dim=1)
        
        # Apply weighted fusion
        fused_feature_map = self.weighted_fusion(fused)
        return fused_feature_map

class EnhancedSEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.color_fusion = ColorChannelFusion(channel=channel, reduction=reduction)
        self.se_attention = ChannelAttention(channel=channel, reduction=reduction)

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
        fused_features = self.color_fusion(x)
        attention_output = self.se_attention(fused_features)
        return attention_output

if __name__ == '__main__':
    model = EnhancedSEAttention()
    model.init_weights()
    input = torch.randn(1, 3, 7, 7)  # Updated to expect 3 channels (R, G, B)
    output = model(input)
    print(output.shape)