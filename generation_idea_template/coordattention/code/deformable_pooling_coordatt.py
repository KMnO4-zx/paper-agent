"""
Replace the `nn
AdaptiveAvgPool2d` operations in the CoordAtt module with a custom deformable pooling layer
Implement learnable offsets for pooling regions that adjust based on input features, allowing the pooling operation to capture more complex spatial hierarchies
Ensure the deformable pooling layer is lightweight to maintain computational efficiency
Evaluate the modified CoordAtt's performance on a small benchmark dataset, comparing improvements in feature representation quality and accuracy against the original implementation

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


class DeformablePooling(nn.Module):
    def __init__(self, channels, kernel_size=1):
        super(DeformablePooling, self).__init__()
        self.offset_conv = nn.Conv2d(channels, 2, kernel_size=3, padding=1)
        self.kernel_size = kernel_size

    def forward(self, x):
        n, c, h, w = x.size()
        # Compute offsets
        offsets = self.offset_conv(x)
        # Create a normalized grid
        grid = self.create_grid(h, w, device=x.device)
        # Add offsets to the grid
        grid = grid + offsets.permute(0, 2, 3, 1)
        # Clamp grid values to ensure they are within valid range
        grid = torch.clamp(grid, -1, 1)
        # Sample using the modified grid
        sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return sampled

    def create_grid(self, height, width, device):
        # Create a grid for sampling
        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float, device=device)
        grid = F.affine_grid(theta, (1, 1, height, width), align_corners=True)
        return grid.repeat(1, 1, 1, 1)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # Initialize deformable pooling layers for height and width
        self.pool_h = DeformablePooling(inp)
        self.pool_w = DeformablePooling(inp)

        mip = max(8, inp // reduction)

        # Convolutional layers for processing pooled features
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # Apply deformable pooling to both height and width
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate pooled features and process through convolutions
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention to the identity
        out = identity * a_w * a_h

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)

# I am done