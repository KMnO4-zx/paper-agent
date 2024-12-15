"""
Modify the `CoordAtt` module
Apply a 1x1 convolution to the *original input feature map* to predict *two sets* of channel-wise scale and bias parameters (outputting four times the number of input channels)
Split the output of the 1x1 convolution into scale and bias parameters for the height features and scale and bias parameters for the width features
*Separately pool the height and width features from the input feature map as before*
Apply the height scale and bias to the *pooled height feature*, and the width scale and bias to the *pooled width feature*, using element-wise multiplication and addition
Then, concatenate the transformed height and width features along the channel dimension
Proceed with the rest of the original `CoordAtt` module's forward pass
In the `__init__` function, add a 1x1 convolution layer for the channel-wise affine transformation parameter prediction, that takes the *original input* (outputting four times the number of input channels)
In the `forward` function, implement the scale and bias parameter prediction from the input, the splitting of scale and bias parameters, the separate pooling of height and width features, the application of separate scale and bias parameters to the pooled height and width features respectively, the concatenation of the transformed height and width features, and then proceed with the rest of the forward pass
Compare output with the baseline using the same test input, and observe changes

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

        # Add 1x1 conv for affine transform parameters
        self.affine_conv = nn.Conv2d(inp, inp * 4, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        
        # Predict affine transform parameters
        affine_params = self.affine_conv(x)
        
        # Split affine parameters into scale and bias for h and w
        scale_h, bias_h, scale_w, bias_w = torch.split(affine_params, [c, c, c, c], dim=1)
        
        # Pool height and width features separately
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Apply channel-wise affine transformation to pooled features
        x_h = x_h * scale_h + bias_h  # Apply scale and bias to pooled height feature
        x_w = x_w * scale_w + bias_w  # Apply scale and bias to pooled width feature
        
        # Concatenate transformed features
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