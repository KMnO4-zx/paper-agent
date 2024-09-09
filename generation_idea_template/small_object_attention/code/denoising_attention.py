"""
Integrate a lightweight denoising autoencoder within the SEAttention framework
Implement an encoder-decoder structure focused on feature compression and noise reduction
Modify the forward function to pass input through the autoencoder before applying channel attention
Optimize the autoencoder's parameters using a transfer learning approach, ensuring it is tailored for small target detection
Evaluate performance by comparing detection accuracy and attention map clarity on small targets with and without the denoising mechanism, using synthetic datasets

"""

# Modified code

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class DenoisingAutoencoder(nn.Module):
    def __init__(self, channel=512, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, channel // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.denoising_autoencoder = DenoisingAutoencoder(channel)
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
        # Pass input through the denoising autoencoder
        denoised = self.denoising_autoencoder(x)
        
        # SEAttention mechanism
        b, c, _, _ = denoised.size()
        y = self.avg_pool(denoised).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return denoised * y.expand_as(denoised)

if __name__ == '__main__':
    model = SEAttention()
    model.init_weights()
    input = torch.randn(1, 512, 7, 7)
    output = model(input)
    print(output.shape)