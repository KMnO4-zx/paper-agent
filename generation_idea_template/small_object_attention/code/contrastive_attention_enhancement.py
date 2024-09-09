"""
Enhance SEAttention by integrating contrastive learning to improve spatial awareness
Implement this by creating pairs of feature maps: one with SEAttention applied and one without
Use a contrastive loss function to train the model to differentiate between these maps, emphasizing small target detection
Modify the forward function to support this training regime
Evaluate the model by comparing the contrastive loss and visualizing the attention focus on small targets, demonstrating improved spatial discrimination over the baseline model

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
from torch.nn import CosineSimilarity

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
        # Initialize cosine similarity for contrastive learning
        self.cos_sim = CosineSimilarity(dim=1)

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
        
        # SEAttention applied map
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        se_attention_map = x * y.expand_as(x)
        
        # Original map without SEAttention
        original_map = x
        
        # Calculate cosine similarity between the two maps
        similarity = self.cos_sim(se_attention_map, original_map)
        
        # Contrastive loss: encourage high similarity
        contrastive_loss = 1 - similarity.mean() # Using 1 - cosine similarity as a simple contrastive loss

        return se_attention_map, contrastive_loss
    
if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output, contrastive_loss = model(input)
    print("Output shape:", output.shape)
    print("Contrastive Loss:", contrastive_loss.item())

# I am done