"""
Modify the `CoordAtt` module
After pooling height and width features, apply *separate* linear layers to project the pooled height and width features into a common lower-dimensional representation
Apply a *shared* linear layer to project both the projected height and width features to a lower-dimensional representation before cross-attention
Apply a scaled dot-product attention mechanism where the projected height features act as the query, and the projected width features act as both the key and value
Use the projected attended height features to *modulate* the original pooled height features via element-wise multiplication
Then, apply another scaled dot-product attention mechanism where the projected width features act as the query, and the projected height features act as both the key and value
Use the projected attended width features to *modulate* the original pooled width features via element-wise multiplication
Concatenate the modulated height and width features along the channel dimension
Apply a 1x1 convolution to the concatenated features to project them to the original input dimension
Feed the projected features to the shared `conv1`
Modify the `__init__` function to include *three* linear projection layers (two separate and one shared), two scaled dot-product attention mechanisms and a 1x1 convolution layer
Modify the `forward` function to implement the initial separate projection layers, the shared projection layer, the two cross-attention mechanisms, element-wise multiplications, concatenation, the final 1x1 convolution and feeding to shared `conv1`
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe the changes
This involves adding three linear projection layers, two cross-attention modules, a 1x1 convolution, two element-wise multiplications, concatenation, and modifying the forward pass

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

        self.proj_shared = nn.Linear(inp, mip)
        self.proj_h = nn.Linear(mip, mip)
        self.proj_w = nn.Linear(mip, mip)


        self.conv1 = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()


        self.attn_h = ScaledDotProductAttention(mip)
        self.attn_w = ScaledDotProductAttention(mip)

        self.conv_out = nn.Conv2d(mip*2, inp, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h_pooled = self.pool_h(x).squeeze(-1).permute(0,2,1) # n, h, c
        x_w_pooled = self.pool_w(x).squeeze(-2).permute(0,2,1) # n, w, c

        x_h_proj_shared = self.proj_shared(x_h_pooled) # n, h, mip
        x_w_proj_shared = self.proj_shared(x_w_pooled) # n, w, mip

        x_h_proj = self.proj_h(x_h_proj_shared) # n, h, mip
        x_w_proj = self.proj_w(x_w_proj_shared) # n, w, mip


        x_h_attn = self.attn_h(x_h_proj, x_w_proj, x_w_proj) # n, h, mip
        x_w_attn = self.attn_w(x_w_proj, x_h_proj, x_h_proj) # n, w, mip

        x_h_mod = x_h_pooled.permute(0,2,1) * x_h_attn.permute(0,2,1) # n, c, h
        x_w_mod = x_w_pooled.permute(0,2,1) * x_w_attn.permute(0,2,1) # n, c, w


        y = torch.cat([x_h_mod, x_w_mod], dim=1) # n, 2c, h/w


        y = self.conv_out(y.unsqueeze(-1)).squeeze(-1) # n, c, h/w

        y = y.unsqueeze(-1)
        y = self.conv1(y).squeeze(-1)
        y = self.bn1(y)
        y = self.act(y)


        y = y.unsqueeze(-1).unsqueeze(-1)
        y = F.interpolate(y, size=(h,w), mode='bilinear', align_corners=False).squeeze(-1).squeeze(-1)

        out = identity * y

        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, key, value):
        attn_weights = torch.bmm(query, key.transpose(1, 2)) / (self.dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, value)
        return attn_output

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)