"""
Develop a quality assessment module that computes a quality score for each input feature map using metrics like noise level or sharpness
Integrate this module into the SEAttention class, modifying the attention weights based on quality scores
Implement this by adding a quality assessment function and updating the forward pass of SEAttention to apply adaptive modulation of attention weights
Evaluate performance improvements using precision, recall, and F1-score on small target detection tasks, and compare results with the baseline SEAttention model and other enhanced models

"""

# Refined code
import numpy as np
import torch
from torch import nn
from torch.nn import init
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
        # Define a Laplacian kernel for sharpness calculation
        self.laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                                [-1,  8, -1],
                                                [-1, -1, -1]]]], dtype=torch.float32)

    def compute_quality_score(self, x):
        # Apply the Laplacian kernel to compute sharpness
        laplacian = F.conv2d(x, self.laplacian_kernel, padding=1)
        sharpness = laplacian.var(dim=[2, 3], keepdim=True)
        quality_score = torch.sigmoid(sharpness)  # Normalize to [0, 1]
        return quality_score

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
        
        # Compute quality score and adjust attention weights
        quality_score = self.compute_quality_score(x)
        adjusted_y = y * quality_score
        
        return x * adjusted_y.expand_as(x)


if __name__ == '__main__':
    # Initialize the model and weights
    model = SEAttention()
    model.init_weights()

    # Test the model with a random input
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)