"""
Modify the `CoordAtt` module
After pooling the height and width features, apply a *single* 1x1 convolution to each of them
This 1x1 conv will project the pooled features to a lower dimension and also transform them for fusion
Apply the sigmoid activation to the transformed features
Introduce a learnable parameter for each of the sigmoid-activated features
Multiply the sigmoid-activated feature with the learnable parameter
Concatenate the modulated height and width attention maps along the channel dimension
Apply a ReLU activation to the concatenated feature map
Then, apply a final 1x1 convolution to the ReLU activated feature map to create the combined attention map
Use this combined attention map to modulate the input feature map
Modify the `__init__` to include the initial 1x1 convolutions, the final 1x1 convolution for fusion, and the learnable parameters
Modify the `forward` to implement the initial 1x1 convolutions, sigmoid activation, modulation with learnable parameters, concatenation, ReLU activation, the final 1x1 convolution for fusion and modulate the input feature map
The output can be compared to the baseline using the same test input and observing the changes in output

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
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1_h = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv1_w = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)

        self.param_h = nn.Parameter(torch.randn(1, mip, 1, 1) * 0.02)
        self.param_w = nn.Parameter(torch.randn(1, mip, 1, 1) * 0.02)

        self.conv_fusion = nn.Conv2d(mip*2, inp, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv1_h(x_h)
        x_w = self.conv1_w(x_w)
        
        a_h = torch.sigmoid(x_h) * self.param_h
        a_w = torch.sigmoid(x_w) * self.param_w

        y = torch.cat([a_h, a_w], dim=1)
        
        y = self.relu(y)
        y = self.conv_fusion(y)


        out = identity * y

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)