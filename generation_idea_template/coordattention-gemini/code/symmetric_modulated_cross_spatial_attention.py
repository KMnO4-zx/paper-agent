"""
Modify the `CoordAtt` module
After pooling height and width features, apply a shared linear layer to project both the pooled height and width features into a common representation
Apply a scaled dot-product attention mechanism where the projected height features act as the query, and the projected width features act as both the key and value
Apply another shared linear layer to project the attended height features back to the original feature dimension
Use the projected attended height features to *modulate* the original pooled width features via element-wise multiplication
Then, apply another scaled dot-product attention mechanism where the projected width features act as the query, and the projected height features act as both the key and value
Apply another shared linear layer to project the attended width features back to the original feature dimension
Use the projected attended width features to *modulate* the original pooled height features via element-wise multiplication
Concatenate the modulated height and width features along the channel dimension
Feed the concatenated features to the shared `conv1`
Modify the `__init__` function to include four shared linear projection layers, and two scaled dot-product attention mechanisms
Modify the `forward` function to implement the projection layers, the two cross-attention mechanisms, element-wise multiplications, concatenation and feeding to shared `conv1`
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe the changes
This involves adding four shared linear projection layers, two cross-attention modules, two element-wise multiplications, concatenation, and modifying the forward pass

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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = dim ** -0.5

    def forward(self, query, key, value):
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, value)
        return out


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # Shared linear projection layers
        self.proj_h = nn.Linear(inp, mip)
        self.proj_w = nn.Linear(inp, mip)

        # Scaled dot-product attention layers
        self.attn_h = ScaledDotProductAttention(mip)
        self.attn_w = ScaledDotProductAttention(mip)
        
        # Output projection layers
        self.proj_out_h = nn.Linear(mip, inp)
        self.proj_out_w = nn.Linear(mip, inp)

        self.conv1 = nn.Conv2d(2 * inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # Pooled features
        x_h = self.pool_h(x).squeeze(-1)  # N, C, H
        x_w = self.pool_w(x).squeeze(-2)  # N, C, W

        # Linear projections
        proj_h = self.proj_h(x_h.transpose(1,2)).transpose(1,2) # N, mip, H
        proj_w = self.proj_w(x_w.transpose(1,2)).transpose(1,2) # N, mip, W


        # Cross-attention for height
        attn_h = self.attn_h(proj_h, proj_w, proj_w)  # N, mip, H
        attn_h = self.proj_out_h(attn_h.transpose(1,2)).transpose(1,2)  # N, C, H

        # Cross-attention for width
        attn_w = self.attn_w(proj_w, proj_h, proj_h)  # N, mip, W
        attn_w = self.proj_out_w(attn_w.transpose(1,2)).transpose(1,2)  # N, C, W

        # Modulation
        x_h = x_h * attn_w  # N, C, H
        x_w = x_w * attn_h  # N, C, W

        # Concatenation and shared conv
        y = torch.cat([x_h, x_w], dim=1) # N, 2C, H/W
        y = self.conv1(y.unsqueeze(-1).unsqueeze(-1)) # N, mip, 1,1
        y = self.bn1(y) # N, mip, 1,1
        y = self.act(y) # N, mip, 1,1

        a_h, a_w = torch.split(y, [c, c], dim=1) # N, C, 1, 1
        
        a_h = a_h.sigmoid()
        a_w = a_w.sigmoid()

        out = identity * a_w.unsqueeze(-2) * a_h.unsqueeze(-1)

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)