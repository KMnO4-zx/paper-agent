"""
Incorporate a complexity assessment module within the SEAttention framework
Implement a function that calculates a complexity score using simple features like pixel intensity variance or entropy from input data
Modify the forward function of SEAttention to adjust the attention weights using this complexity score
Evaluate the model's performance across small target detection tasks with varying image complexities, using metrics such as precision, recall, and F1-score
Compare the results against the baseline SEAttention model to demonstrate improvements in robustness and adaptability

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
import torchvision.transforms as transforms


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
    
    def complexity_score(self, x):
        # Convert to grayscale for simplicity
        gray_transform = transforms.Grayscale()
        x_gray = gray_transform(x)
        
        # Compute pixel intensity variance as complexity score
        variance = torch.var(x_gray, dim=(2, 3), keepdim=True)
        
        # Normalize the variance to be between 0 and 1
        max_variance = torch.max(variance)
        min_variance = torch.min(variance)
        complexity_score = (variance - min_variance) / (max_variance - min_variance + 1e-5)
        
        return complexity_score

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
        # Calculate complexity score
        complexity = self.complexity_score(x)
        
        # Original SEAttention operations
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Adjust attention weights using complexity score
        adjusted_y = y * complexity
        
        return x * adjusted_y.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)