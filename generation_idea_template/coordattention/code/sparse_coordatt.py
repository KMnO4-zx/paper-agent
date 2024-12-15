"""
Integrate an L1 regularization term into the training process of the CoordAtt module to induce sparsity in the feature maps
Modify the loss function to include this L1 penalty, encouraging sparsity in the output of the initial convolutional layers
Evaluate the impact on feature representation quality and computational efficiency by testing on a small benchmark dataset
Compare the results in terms of accuracy, feature discrimination, and computational overhead with the original CoordAtt and other variants

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

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

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

def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

# Example training loop
def train(model, dataloader, criterion, optimizer, lambda_l1):
    model.train()
    total_loss = 0.0
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Add L1 regularization penalty
        l1_penalty = l1_regularization(model, lambda_l1)
        loss += l1_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    att = CoordAtt(inp=64, reduction=32)
    out = att(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)

    # Example of lambda_l1 for L1 regularization
    lambda_l1 = 0.01

    # Mock dataloader, criterion, and optimizer for testing
    dataloader = [ (x, torch.randn(2, 64, 32, 32)) for _ in range(10) ]
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(att.parameters(), lr=0.01)

    # Run a single training epoch
    avg_loss = train(att, dataloader, criterion, optimizer, lambda_l1)
    print("Average training loss with L1 regularization:", avg_loss)