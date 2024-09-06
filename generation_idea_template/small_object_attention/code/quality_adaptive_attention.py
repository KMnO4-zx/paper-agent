"""
Develop a quality assessment module that computes a quality score for each input feature map using metrics like noise level or sharpness
Integrate this module into the SEAttention class, modifying the attention weights based on quality scores
Implement this by adding a quality assessment function and updating the forward pass of SEAttention to apply adaptive modulation of attention weights
Evaluate performance improvements using precision, recall, and F1-score on small target detection tasks, and compare results with the baseline SEAttention model and other enhanced models

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

    def compute_quality_score(self, x):
        # Example quality assessment using noise level (variance) and sharpness (Laplacian)
        noise_level = torch.var(x, dim=(2, 3), keepdim=True)
        laplacian = torch.nn.functional.conv2d(
            x, weight=torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=x.dtype, device=x.device),
            padding=1
        )
        sharpness = torch.mean(torch.abs(laplacian), dim=(2, 3), keepdim=True)
        quality_score = 1.0 / (1.0 + noise_level) * (1.0 + sharpness)
        return quality_score

    def forward(self, x):
        b, c, _, _ = x.size()
        quality_score = self.compute_quality_score(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        adaptive_weights = y * quality_score
        return x * adaptive_weights.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)