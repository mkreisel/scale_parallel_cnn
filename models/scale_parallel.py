import torch.nn as nn
import torch.nn.functional as F

class DownUpBlock(nn.Module):
    def __init__(self, ni, nf, size, stride=1, drop_p=0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = nn.Conv2d(ni, nf, size, stride)
        self.deconv1 = nn.ConvTranspose2d(ni, nf, size, stride)
        self.drop = nn.Dropout(drop_p, inplace=True) if drop_p else None

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.deconv1(x), inplace=True)
        if self.drop: x = self.drop(x)
        x = self.bn(x)
        return x

class ScaleParallelBlock(nn.Module):
    def __init__(self, ni, nf, sizes, stride, drop_p=0.0):
        super().__init__()
        self.conv_layers = [DownUpBlock(ni, nf, sizes[i], stride, drop_p) for i in range(len(sizes))]

    def forward(self, x):
        outputs = [layer(x) for layer in self.conv_layers]
        return torch.cat(outputs)

class ScaleParallelNet1(nn.Module):
    def __init__(self, ni=1, nf=20, sizes=[4,8,12], stride=1, drop_p=0.0):
        super().__init__()
        self.parallel_block = ScaleParallelBlock(ni, nf, sizes, stride, drop_p)
        self.conv1x1 = nn.Conv2d(60, 10, 1, 1)
        self.linear = nn.Linear(10240, 10)

    def forward(self, x):
        x = self.parallel_block(x)
        x = self.conv1x1(x)
        x = F.relu(x, inplace=True)
        x = x.view(-1, 10240)
        x = self.linear(x)
        return x