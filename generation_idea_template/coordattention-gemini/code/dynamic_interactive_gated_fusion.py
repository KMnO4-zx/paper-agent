"""
Modify the `CoordAtt` module
After pooling height and width features, apply a 1x1 convolution to each of them *separately*, outputting the same number of channels as the input
Concatenate the transformed height and width features along the channel dimension
Then, apply a *single 1x1 convolution* to the concatenated features, outputting *twice* the number of channels as the individual transformed height or width features
Apply a sigmoid activation to the output of this 1x1 conv
Split the output into two equal parts, representing attention scores for height and width respectively
Modulate the transformed pooled height and width features using their respective attention scores via element-wise multiplication
Concatenate the modulated height and width features along the channel dimension
Feed the concatenated features to the shared `conv1`
In the `__init__` function, add *two 1x1 convolution layers* for the initial transformation, outputting the same number of channels as the input and *one 1x1 convolution layer* for generating attention scores, outputting *twice* the number of channels as the individual transformed height or width features
In the `forward` function, implement the initial 1x1 convolutions, the concatenation, the application of the attention generation 1x1 conv followed by sigmoid, the splitting of the attention scores, the modulation of the pooled height and width features using the attention scores, the concatenation, and feeding to the shared `conv1`
The rest of the forward pass remains the same
Compare output with the baseline using the same test input and observe the changes
This involves modifying `__init__` to include the three 1x1 conv layers, and `forward` to implement the feature transformation, concatenation, attention score generation, splitting, modulation, and final concatenation

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

        # mip = max(8, inp // reduction) # Not needed anymore

        # Added 1x1 conv layers for feature transformation
        self.conv_h_transform = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w_transform = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)

        # Added 1x1 conv layer for attention score generation
        self.conv_attention = nn.Conv2d(2 * inp, 2 * inp, kernel_size=1, stride=1, padding=0)


        # self.conv1 = nn.Conv2d(2 * inp, mip, kernel_size=1, stride=1, padding=0) # Not needed anymore
        # self.bn1 = nn.BatchNorm2d(mip) # Not needed anymore
        # self.act = h_swish() # Not needed anymore



    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Apply 1x1 conv for feature transformation
        x_h_transformed = self.conv_h_transform(x_h)
        x_w_transformed = self.conv_w_transform(x_w)

        # Concatenate transformed features
        y = torch.cat([x_h_transformed, x_w_transformed], dim=1)


        # Generate attention scores
        attention_scores = self.conv_attention(y).sigmoid()
        
        # Split attention scores
        a_h, a_w = torch.split(attention_scores, [inp, inp], dim=1)


        # Modulate transformed pooled features
        x_h_modulated = x_h_transformed * a_h
        x_w_modulated = x_w_transformed * a_w

        # Upsample modulated features to original input size
        x_h_modulated = F.interpolate(x_h_modulated, size=(h, w), mode='bilinear', align_corners=False)
        x_w_modulated = F.interpolate(x_w_modulated.permute(0, 1, 3, 2), size=(h, w), mode='bilinear', align_corners=False)



        # Combine modulated features
        out = identity * (x_h_modulated + x_w_modulated)


        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)