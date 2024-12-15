"""
Modify the `CoordAtt` module
After pooling height and width features, apply a *shared* 1x1 convolution to both the pooled height and width features to project them into a common dimension
Apply a ReLU activation to the projected features
Use the projected height feature as the query and the projected width feature as the key and value for a dot-product attention mechanism
Introduce a *learnable bias parameter* that is *added* to the dot-product attention map before the softmax operation
Apply another *shared* 1x1 convolution to the output of the attention
Modulate the *original* pooled height feature with the output of the attention via element-wise multiplication
Repeat the same process with the projected width as the query and projected height as key/value
Then, modulate the *original* pooled width feature with the output of the attention via element-wise multiplication
Concatenate the modulated height and width features along the channel dimension
Feed the concatenated features to the shared `conv1`
In the `__init__` function, add *two shared* 1x1 convolution layers for the projection and post attention projection, a ReLU activation, and *a learnable bias parameter*
In the `forward` function, implement the projection, ReLU activation, the dot-product attention mechanism, the addition of the learnable bias parameter to the attention map, the post attention projection, the modulation of the original pooled features with the attention output, the concatenation, and feeding to the shared `conv1`
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe the changes
This involves adding two shared 1x1 convs, implementing ReLU activation, dot-product attention, learnable bias parameter, post attention projection, modulation of pooled features with attention, concatenation and modifying the forward pass

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

        # Shared 1x1 conv for projection
        self.proj_conv = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # ReLU activation
        self.relu = nn.ReLU()
        # Learnable bias parameter
        self.bias = nn.Parameter(torch.zeros(1))
         # Shared 1x1 conv for post-attention projection
        self.post_attn_conv = nn.Conv2d(mip, mip, kernel_size=1, stride=1, padding=0)


        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Project pooled features
        proj_h = self.proj_conv(x_h)
        proj_w = self.proj_conv(x_w)
        
        # Apply ReLU
        proj_h = self.relu(proj_h)
        proj_w = self.relu(proj_w)


        # Dot-product attention with learnable bias
        attn_hw = torch.matmul(proj_h.transpose(-2,-1), proj_w) # n, mip, h, w
        attn_hw = attn_hw + self.bias
        attn_hw = F.softmax(attn_hw, dim=-1)

        attn_wh = torch.matmul(proj_w.transpose(-2,-1), proj_h) # n, mip, w, h
        attn_wh = attn_wh + self.bias
        attn_wh = F.softmax(attn_wh, dim=-1)


        # Post attention projection
        attn_hw_out = self.post_attn_conv(torch.matmul(proj_w, attn_hw.transpose(-2,-1))) # n, mip, h, 1
        attn_wh_out = self.post_attn_conv(torch.matmul(proj_h, attn_wh.transpose(-2,-1))) # n, mip, w, 1

        # Modulate original pooled features
        x_h_modulated = x_h * attn_hw_out
        x_w_modulated = x_w * attn_wh_out
        

        # Concatenate modulated features
        y = torch.cat([x_h_modulated, x_w_modulated.permute(0,1,3,2)], dim=2)


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