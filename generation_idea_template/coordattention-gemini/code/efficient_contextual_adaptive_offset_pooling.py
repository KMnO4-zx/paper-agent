"""
Modify the `CoordAtt` module
Introduce a 1x1 convolutional layer that takes the input feature map and outputs a contextual feature map
Then, introduce another 1x1 convolutional layer that takes this contextual feature map and outputs two sets of spatial offset maps, one for height pooling and another for width pooling
The spatial dimensions of the offset maps should match the corresponding dimensions of the input feature map
Apply a tanh activation to the output of the 1x1 convolution used to generate the offset maps, constraining the offset to a small range (e
g
, [-1, 1])
During the pooling operation (both height and width), use these generated offsets to sample from the input feature map using bilinear interpolation
This means that the pooling operation will be performed on features sampled from the input at shifted locations according to the predicted offset
The rest of the forward pass remains unchanged
In the `__init__` function, add the two 1x1 convolutional layers and the tanh activation
In the `forward` function, implement the generation of the contextual feature map, the generation of spatial offset maps, the application of the tanh activation, and modify the pooling to incorporate these offsets using bilinear interpolation
Compare the output with the baseline using the same test input to observe changes
This requires adding two 1x1 conv layers, applying tanh activation, generating spatial offset maps from a contextual representation, implementing bilinear sampling and modifying the pooling operation in the forward pass

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

        # New layers for offset prediction
        self.context_conv = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.offset_conv = nn.Conv2d(mip, 2, kernel_size=1, stride=1, padding=0)  # 2 channels for height and width offsets
        self.tanh = nn.Tanh()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        
        # Generate contextual feature map
        context = self.context_conv(x)
        
        # Generate spatial offset maps
        offset = self.offset_conv(context)
        offset = self.tanh(offset)

        offset_h = offset[:, 0:1, :, :]  # Offset for height pooling, shape [n, 1, h, w]
        offset_w = offset[:, 1:2, :, :]  # Offset for width pooling, shape [n, 1, h, w]

        # Generate sampling grid for height pooling
        h_indices = torch.arange(0, h, dtype=torch.float32, device=x.device).view(1, 1, h, 1).repeat(n, 1, 1, w)
        h_indices = h_indices + offset_h * 0.5 * (h-1) # Scale the offset to the range [-0.5, 0.5]

        # Ensure indices within bounds
        h_indices = torch.clamp(h_indices, 0, h-1)

        # Generate sampling grid for width pooling
        w_indices = torch.arange(0, w, dtype=torch.float32, device=x.device).view(1, 1, 1, w).repeat(n, 1, h, 1)
        w_indices = w_indices + offset_w * 0.5 * (w-1) # Scale the offset to the range [-0.5, 0.5]

        # Ensure indices within bounds
        w_indices = torch.clamp(w_indices, 0, w-1)

        # Construct grid for height pooling
        grid_h_x = torch.zeros_like(w_indices)
        grid_h_y = h_indices / (h - 1) * 2 - 1
        grid_h = torch.cat([grid_h_x, grid_h_y], dim=1).permute(0, 2, 3, 1)

        # Construct grid for width pooling
        grid_w_x = w_indices / (w - 1) * 2 - 1
        grid_w_y = torch.zeros_like(h_indices)
        grid_w = torch.cat([grid_w_x, grid_w_y], dim=1).permute(0, 2, 3, 1)


        # Perform bilinear sampling for height pooling
        sampled_h = F.grid_sample(x, grid_h, mode='bilinear', align_corners=True)
        x_h = self.pool_h(sampled_h)

        # Perform bilinear sampling for width pooling
        sampled_w = F.grid_sample(x, grid_w, mode='bilinear', align_corners=True)
        x_w = self.pool_w(sampled_w).permute(0, 1, 3, 2)

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