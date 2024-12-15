"""
Modify the `CoordAtt` module
After the height and width pooling operations, concatenate the pooled height and width features along the channel dimension
Apply a 1x1 convolution to reduce the number of channels to the original input size
Apply another 1x1 convolution that outputs two sets of weights, each with the same dimension as the reduced feature map
Apply a sigmoid activation to these weights
Apply a 1x1 convolution to the original concatenated features to project them to the same dimension as the reduced feature map
Modulate the reduced feature map using the first set of weights
Modulate the projected original concatenated features using the second set of weights
Add the two modulated features
Apply a single depthwise separable convolution to the result
Split the output of the depthwise separable convolution into two equal parts representing height and width features respectively
Then, feed these features to the shared `conv1`
Modify the `__init__` function to include the three 1x1 convolution layers for the gated fusion, and a single depthwise separable convolution block
Modify the `forward` function to implement the concatenation of the pooled features, the 1x1 convolution for channel reduction, the 1x1 convolution for projecting the original concatenated features, the gated fusion, the depthwise separable convolution, and the splitting of the output before feeding it to the shared conv1
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe changes
This involves adding three 1x1 conv layers for gated fusion, a single depthwise separable conv block, concatenation and splitting, and modifying the forward pass to apply this before shared conv1

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

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv_reduce = nn.Conv2d(inp * 2, mip, kernel_size=1, stride=1, padding=0) # 1x1 conv for channel reduction after concatenation
        self.conv_gate = nn.Conv2d(mip, mip * 2, kernel_size=1, stride=1, padding=0) # 1x1 conv for generating gating weights
        self.conv_proj = nn.Conv2d(inp * 2, mip, kernel_size=1, stride=1, padding=0) # 1x1 conv for projecting original concatenated feature

        self.depthwise_conv = DepthwiseSeparableConv(mip, mip)

        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv1 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2) # concatenate pooled features

        y_reduced = self.conv_reduce(y) # channel reduction
        
        gates = self.conv_gate(y_reduced) # generating gating weights
        gate1, gate2 = torch.split(gates, gates.shape[1] // 2, dim=1)
        gate1 = gate1.sigmoid()
        gate2 = gate2.sigmoid()

        y_proj = self.conv_proj(y) # projection of original concatenated features

        y_reduced_mod = y_reduced * gate1
        y_proj_mod = y_proj * gate2

        y_fused = y_reduced_mod + y_proj_mod

        y_fused = self.depthwise_conv(y_fused)

        x_h, x_w = torch.split(y_fused, [h, w], dim=2) # split back into height and width
        x_w = x_w.permute(0, 1, 3, 2)



        y = self.bn1(self.act(x_h+x_w))
        y = self.conv1(y)
       

        out = identity * y

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)