"""
Modify the CoordAtt module to incorporate both spatial and frequency domain attention mechanisms
Perform a Fast Fourier Transform (FFT) on the input feature maps to capture frequency domain information
Compute attention weights separately for spatial and frequency domains, then merge them using a simple weighted sum or concatenation followed by a linear transformation
Adjust the forward method to include these steps, and evaluate the module's performance on a small benchmark dataset, comparing improvements in feature representation, accuracy, and computational efficiency against the original CoordAtt and other variants

"""

# Modified code
import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32, freq_weight=0.5):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Frequency domain attention
        self.freq_weight = freq_weight
        self.conv_freq = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn_freq = nn.BatchNorm2d(mip)

    def forward(self, x):
        identity = x

        # Spatial attention
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y_spatial = torch.cat([x_h, x_w], dim=2)
        y_spatial = self.conv1(y_spatial)
        y_spatial = self.bn1(y_spatial)
        y_spatial = self.act(y_spatial)

        x_h, x_w = torch.split(y_spatial, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Frequency domain attention
        x_freq = torch.fft.fft2(x)
        x_freq = torch.abs(x_freq)  # Use magnitude spectrum
        x_freq = self.conv_freq(x_freq)
        x_freq = self.bn_freq(x_freq)
        x_freq = self.act(x_freq)

        # Weighted sum of spatial and frequency domain attention
        out = (1 - self.freq_weight) * identity * a_w * a_h + self.freq_weight * x_freq

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)