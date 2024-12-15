"""
Modify the `CoordAtt` module
After pooling height and width features, apply a *separate* 1x1 convolution to each of them, projecting them to the original input channel size
Concatenate the *projected* height and width features along the channel dimension
Apply a squeeze-and-excitation (SE) block to the concatenated features
The SE block will consist of global average pooling, followed by a 1x1 convolution with ReLU activation, and another 1x1 convolution with sigmoid activation
Split the output of the SE block into two equal parts, representing attention scores for height and width respectively
Use these attention scores to modulate (element-wise multiply) the *original* pooled height and width features separately
Then, project these modulated height and width features to the original channel dimension using separate 1x1 convolutions (`conv_h` and `conv_w` are used here)
In the `__init__` function, add *two* 1x1 convolution layers for projecting pooled height and width features to the original channel size, and the SE block (global average pooling, two 1x1 convs, ReLU and Sigmoid)
In the `forward` function, implement the initial 1x1 projections, concatenation, the SE block, splitting of the SE output into height and width attention scores, modulation of the original pooled height and width features using their respective attention scores, and then pass these modulated features to `conv_h` and `conv_w`
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe changes
This involves adding two 1x1 convs for projecting, the SE block, and modifying the forward pass to implement the projection, concatenation, SE, splitting, and modulation before `conv_h` and `conv_w`

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

        # Projection layers for pooled h and w
        self.proj_h = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.proj_w = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)

        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2 * inp, mip, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, 2 * inp, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_h = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Project pooled features
        proj_x_h = self.proj_h(x_h)
        proj_x_w = self.proj_w(x_w)


        # Concatenate projected features
        y = torch.cat([proj_x_h, proj_x_w.permute(0,1,3,2)], dim=1)

        # SE block
        se_out = self.se(y)

        # Split SE output into attention scores
        a_h, a_w = torch.split(se_out, [c, c], dim=1)

        # Modulate original pooled features
        x_h = x_h * a_h
        x_w = x_w * a_w.permute(0,1,3,2)



        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w).permute(0,1,3,2)


        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)