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
from torch.distributions import Categorical

class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MonteCarloDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

class UncertaintyEstimator(nn.Module):
    def __init__(self, channel, num_samples=10):
        super(UncertaintyEstimator, self).__init__()
        self.num_samples = num_samples
        self.dropout = MonteCarloDropout(p=0.5)
        self.conv = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        # Use Monte Carlo sampling to estimate uncertainty
        predictions = torch.stack([self.conv(self.dropout(x)) for _ in range(self.num_samples)], dim=0)
        mean_prediction = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0).mean(dim=(2, 3), keepdim=True)  # Calculate uncertainty as variance
        return mean_prediction, uncertainty

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
        self.uncertainty_estimator = UncertaintyEstimator(channel)

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
        
        # Estimate uncertainty
        mean_prediction, uncertainty = self.uncertainty_estimator(x)
        
        # Integrate uncertainty into attention weights
        # Here, uncertainty is used to scale the attention weights, 
        # with higher uncertainty leading to lower attention weights.
        attention = x * y.expand_as(x)
        adjusted_attention = attention * (1 - uncertainty)
        
        return adjusted_attention

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)