"""
Integrate a frequency domain analysis step within the CoordAtt module
Implement a function to perform Fast Fourier Transform (FFT) on the input feature maps, focusing on extracting significant frequency components
Use these components to modulate the attention weights in the CoordAtt module
Modify the forward method to incorporate this frequency domain information before applying the spatial attention mechanism
Evaluate the impact on feature representation using metrics such as accuracy, feature representation quality, and computational efficiency by testing on a small benchmark dataset
Compare the performance to the original CoordAtt and other variants

"""
### aaa
# Modified code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

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
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        # Compute FFT on a downsampled version of input feature maps
        pool_x = F.adaptive_avg_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        fft_x = torch.fft.fft2(pool_x)
        fft_x = torch.fft.fftshift(fft_x)

        # Extract significant frequency components
        freq_magnitude = torch.abs(fft_x)

        # Normalize frequency components to modulate attention
        freq_magnitude = (freq_magnitude - freq_magnitude.min()) / (freq_magnitude.max() - freq_magnitude.min())

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Modulate attention weights with frequency magnitude
        a_h = a_h * F.interpolate(freq_magnitude[:, :, :h, :], size=(h, 1))
        a_w = a_w * F.interpolate(freq_magnitude[:, :, :, :w], size=(1, w))

        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)