"""
Integrate a temporal attention mechanism using LSTM or GRU layers into the SEAttention framework
This involves processing sequences of input frames to generate a temporal attention map
Combine the temporal map with the SEAttention channel output through element-wise multiplication
Evaluate improvements in detection metrics such as precision, recall, and F1-score on small target detection tasks, comparing against the baseline SEAttention model

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

    def __init__(self, channel=512,reduction=16):
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

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TemporalAttention(nn.Module):
    
    def __init__(self, channel=512, hidden_size=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=channel, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, channel)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, t, c, h, w = x.size()  # assuming input shape is (batch, time, channel, height, width)
        x = x.view(b, t, c * h * w)  # flatten spatial dimensions
        _, h_n = self.gru(x)  # h_n is the last hidden state
        y = self.fc(h_n[-1])  # take the last layer's hidden state
        y = self.sigmoid(y).view(b, c, 1, 1)
        return y

class SEAttentionWithTemporal(nn.Module):

    def __init__(self, channel=512, reduction=16, hidden_size=256, num_layers=1):
        super().__init__()
        self.se_attention = SEAttention(channel, reduction)
        self.temporal_attention = TemporalAttention(channel, hidden_size, num_layers)

    def forward(self, x):
        se_output = self.se_attention(x[:, -1])  # apply SEAttention on the last frame
        temporal_map = self.temporal_attention(x)
        return se_output * temporal_map.expand_as(se_output)

if __name__ == '__main__':
    model = SEAttentionWithTemporal()
    model.se_attention.init_weights()
    input = torch.randn(1, 5, 512, 7, 7)  # example with 5-frame sequence
    output = model(input)
    print(output.shape)