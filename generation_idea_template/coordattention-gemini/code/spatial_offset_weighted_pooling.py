"""
Modify the `CoordAtt` module
Introduce a small convolutional network (e
g
, a single 1x1 conv layer) to generate spatial offset maps for height and width, respectively, from the input feature map
This conv layer will output two sets of channels: one set for height offset maps with the same spatial size as the height dimension, and another for width offset maps with the same spatial size as the width dimension
Constrain the offset maps to be within a small range (e
g
, [-1, 1])
Before pooling, calculate the offset indices by adding the feature-dependent spatial offset map to the original index, where each position in the spatial dimension has a different offset
For each pooled position, perform a weighted average of the values at the immediate integer indices surrounding the offset index, using the fractional part of the offset as weights
Modify the `__init__` to include the 1x1 conv layer for generating the spatial offset maps, and constrain the output to a reasonable range
Modify the `forward` function to generate the spatial feature-dependent offsets, calculate the offset indices, perform the weighted pooling using the calculated offset maps, and use the resulting feature maps for the rest of the forward pass
The rest of the forward pass remains the same
Compare the output with the baseline using the same test input and observe changes
This involves adding a 1x1 conv layer, generating spatially varying feature-dependent offset maps, implementing offset index calculation, weighted pooling, and modifying the pooling operation in the forward pass

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
    def __init__(self, inp, reduction=32, offset_h_channels=8, offset_w_channels=8, offset_scale=0.5):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Add 1x1 conv for offset maps
        self.offset_conv_h = nn.Conv2d(inp, offset_h_channels, kernel_size=1, stride=1, padding=0)
        self.offset_conv_w = nn.Conv2d(inp, offset_w_channels, kernel_size=1, stride=1, padding=0)
        self.offset_scale = offset_scale
        self.offset_h_channels = offset_h_channels
        self.offset_w_channels = offset_w_channels


    def _get_offset_indices(self, size, offset_map, dim):
        """Calculates offset indices."""
        n, c, h, w = offset_map.size()
        if dim == 2:  # height
            offset_map = offset_map.view(n, self.offset_h_channels, size, 1)
        elif dim == 3:  # width
           offset_map = offset_map.view(n, self.offset_w_channels, 1, size)
        else:
             raise ValueError("Invalid dimension for offset calculation")
        offset_map = self.offset_scale * offset_map
        
        
        index = torch.arange(size, device=offset_map.device, dtype=torch.float32)
        if dim == 2:
            index = index.view(1, 1, size, 1)
        elif dim == 3:
            index = index.view(1, 1, 1, size)
        
        offset_index = index + offset_map

        floor_index = torch.floor(offset_index).long()
        ceil_index = torch.ceil(offset_index).long()

        floor_index = torch.clamp(floor_index, 0, size - 1)
        ceil_index = torch.clamp(ceil_index, 0, size - 1)

        offset_weight = offset_index - floor_index.float()
        
        return floor_index, ceil_index, offset_weight
    

    def _weighted_pool(self, x, offset_map, dim):
      """Performs weighted average pooling with spatial offsets."""
      n, c, h, w = x.size()
      if dim == 2:
          size = h
          floor_index, ceil_index, offset_weight = self._get_offset_indices(size, offset_map, dim)
          gathered_x = x.gather(2, floor_index.expand(-1,c,-1,-1)) * (1-offset_weight) + x.gather(2, ceil_index.expand(-1,c,-1,-1)) * offset_weight
      elif dim == 3:
          size = w
          floor_index, ceil_index, offset_weight = self._get_offset_indices(size, offset_map, dim)
          gathered_x = x.gather(3, floor_index.expand(-1,c,-1,-1)) * (1-offset_weight) + x.gather(3, ceil_index.expand(-1,c,-1,-1)) * offset_weight
      else:
          raise ValueError("Invalid dimension for pooling")
      
      return gathered_x
    


    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Generate spatial offsets
        offset_h = self.offset_conv_h(x)
        offset_w = self.offset_conv_w(x)

        # Weighted pooling with offsets
        x_h = self._weighted_pool(x, offset_h, dim=2).mean(dim=1, keepdim=True) # Pooling along height
        x_w = self._weighted_pool(x, offset_w, dim=3).mean(dim=1, keepdim=True).permute(0, 1, 3, 2) # Pooling along width

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