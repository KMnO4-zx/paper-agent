"""
Modify the `CoordAtt` module
Introduce a 1x1 convolution layer to predict initial spatial kernels for height and width modulation
The output of this 1x1 convolution should have a number of channels equal to the input channels * 2
In the forward pass, split the channels into initial height and width modulation kernels of the same size as input
Concatenate the initial height and width kernels along the channel dimension
Apply a lightweight spatial attention module to the concatenated kernels
The spatial attention module will consist of a 3x3 depthwise convolution, followed by a 1x1 convolution and finally a sigmoid activation
Split the output of the spatial attention module back into refined height and width kernels
Use the spatially refined height kernel to modulate the input feature map before height pooling via element-wise multiplication
Similarly, use the spatially refined width kernel to modulate the input feature map before width pooling via element-wise multiplication
Then perform standard average pooling
The rest of the forward pass remains the same
Modify the `__init__` to include the 1x1 convolution for initial kernel prediction and the spatial attention module
Modify the `forward` to implement the concatenation of initial kernels, the spatial attention on the concatenated kernels, the splitting of the output, and then use them for pre-pooling modulation
Compare the output with the baseline using the same test input and observe changes
This involves adding a 1x1 conv layer, spatial attention module, generating initial spatial kernels, concatenating them, applying spatial attention, splitting them and modifying the forward pass to use them for pre-pooling modulation

"""

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

        # 1x1 conv for initial kernel prediction
        self.initial_conv = nn.Conv2d(inp, inp * 2, kernel_size=1, stride=1, padding=0)

        # Spatial attention module
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2*inp, 2*inp, kernel_size=3, stride=1, padding=1, groups=2*inp), # Depthwise conv
            nn.Conv2d(2*inp, 2*inp, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        
        # Initial spatial kernel prediction
        initial_kernels = self.initial_conv(x)
        initial_h_kernel, initial_w_kernel = torch.split(initial_kernels, [c, c], dim=1)

        # Concatenate initial kernels
        concatenated_kernels = torch.cat([initial_h_kernel, initial_w_kernel], dim=1)

        # Spatial attention
        refined_kernels = self.spatial_attn(concatenated_kernels)

        # Split refined kernels
        refined_h_kernel, refined_w_kernel = torch.split(refined_kernels, [c, c], dim=1)

        # Pre-pooling modulation
        x_h_modulated = x * refined_h_kernel
        x_w_modulated = x * refined_w_kernel
        
        x_h = self.pool_h(x_h_modulated)
        x_w = self.pool_w(x_w_modulated).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)