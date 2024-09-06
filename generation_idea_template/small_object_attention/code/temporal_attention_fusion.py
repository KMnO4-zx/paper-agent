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

class SEAttentionWithTemporal(nn.Module):

    def __init__(self, channel=512, reduction=16, lstm_hidden_size=128):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(input_size=channel, hidden_size=lstm_hidden_size, batch_first=True)
        self.temporal_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, channel),
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

    def forward(self, x_seq):
        # x_seq is expected to have shape (batch, seq_len, channel, height, width)
        b, t, c, h, w = x_seq.size()
        x_seq = x_seq.view(b * t, c, h, w)
        
        # SE Attention
        y = self.avg_pool(x_seq).view(b * t, c)
        y = self.fc(y).view(b, t, c, 1, 1)
        se_output = x_seq.view(b, t, c, h, w) * y.expand_as(x_seq.view(b, t, c, h, w))
        
        # Temporal Attention
        temporal_input = se_output.view(b, t, c, -1).mean(-1)  # Reduce spatial dimensions
        temporal_output, _ = self.lstm(temporal_input)
        temporal_weights = self.temporal_fc(temporal_output).view(b, t, c, 1, 1)
        
        # Element-wise multiplication of SE and Temporal Attention outputs
        final_output = se_output * temporal_weights.expand_as(se_output)
        return final_output
    
if __name__ == '__main__':
    model = SEAttentionWithTemporal()
    model.init_weights()
    input_seq = torch.randn(1, 5, 512, 7, 7)  # Example input with sequence length 5
    output = model(input_seq)
    print(output.shape)