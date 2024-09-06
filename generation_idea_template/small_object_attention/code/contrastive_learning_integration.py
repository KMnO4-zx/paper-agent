"""
Integrate a contrastive learning module into the SEAttention framework
Develop functions to generate contrastive pairs from input data, either through augmentations or synthetic data creation
Ensure these pairs highlight small target presence or absence
Modify the training loop to include a contrastive loss alongside the standard detection loss
Evaluate the performance improvements using metrics such as precision, recall, and F1-score on small target detection tasks, comparing results with the baseline SEAttention model and other enhanced models
Emphasize robustness in varied detection scenarios

"""

# Modified code
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
from torchvision import transforms

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

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ContrastiveLearningModule(nn.Module):
    
    def __init__(self, feature_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim, bias=False)
        )
    
    def forward(self, x1, x2):
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        return z1, z2

def contrastive_loss(z1, z2, temperature=0.5, device='cpu'):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(device)
    similarity_matrix = torch.matmul(z1, z2.T) / temperature
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def generate_contrastive_pairs(input_data):
    # Applying random augmentations to generate pairs
    transform = transforms.Compose([
        transforms.RandomResizedCrop(7),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    augmented_data_1 = transform(input_data)
    augmented_data_2 = transform(input_data)
    return augmented_data_1, augmented_data_2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEAttention().to(device)
    model.init_weights()
    contrastive_model = ContrastiveLearningModule(feature_dim=512).to(device)

    input_data = torch.randn(10, 512, 7, 7).to(device)  # Example for a batch size of 10
    output = model(input_data)
    
    augmented_data_1, augmented_data_2 = generate_contrastive_pairs(input_data)
    z1, z2 = contrastive_model(output.view(output.size(0), -1), output.view(output.size(0), -1))
    cl_loss = contrastive_loss(z1, z2, device=device)
    
    print(f'Output shape: {output.shape}, Contrastive Loss: {cl_loss.item()}')