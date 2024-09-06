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
from torch.nn.functional import adaptive_avg_pool2d

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

    def complexity_score(self, x):
        # Calculate pixel intensity variance as a measure of complexity
        variance = torch.var(x, dim=(2, 3), keepdim=True)
        normalized_variance = variance / (torch.mean(variance) + 1e-5)
        return normalized_variance

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Integrate complexity score
        complexity_score = self.complexity_score(x)
        adjusted_attention = y * (1 + complexity_score)

        return x * adjusted_attention.expand_as(x)
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)