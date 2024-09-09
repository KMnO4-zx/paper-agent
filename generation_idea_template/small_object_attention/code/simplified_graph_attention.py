"""
Extend SEAttention by incorporating a simplified Graph Neural Network (GNN) layer
Treat feature maps as graphs with nodes representing spatial locations and edges encoding basic spatial relationships or proximity
Implement a lightweight graph convolution technique to process these graphs, focusing on essential spatial dependencies
Integrate this GNN layer after the channel attention stage
Modify the forward function to include basic graph construction and processing
Evaluate the model's performance on synthetic datasets by comparing detection accuracy and attention focus against baseline SEAttention, with emphasis on capturing spatial dependencies efficiently

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
from torch_geometric.nn import GCNConv  # Importing graph convolutional layer

class SEAttentionGNN(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Define a GCN layer for processing the graph
        self.gcn = GCNConv(channel, channel)

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
        
        # Channel attention
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        
        # Convert feature maps to graph
        x_flat = x.view(b, c, -1).permute(0, 2, 1)  # Reshape to (b, h*w, c)
        
        # Create adjacency matrix using spatial proximity (considering 4-connectivity)
        edge_index = []
        for i in range(h):
            for j in range(w):
                index = i * w + j
                if i + 1 < h:  # Down
                    edge_index.append([index, (i + 1) * w + j])
                if j + 1 < w:  # Right
                    edge_index.append([index, i * w + (j + 1)])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Apply GCN layer
        x_graph = []
        for i in range(b):
            x_graph.append(self.gcn(x_flat[i], edge_index))
        
        x_graph = torch.stack(x_graph).permute(0, 2, 1).view(b, c, h, w)
        
        return x_graph

if __name__ == '__main__':
    model = SEAttentionGNN()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)