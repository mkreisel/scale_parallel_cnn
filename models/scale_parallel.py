import torch.nn as nn
import torch.nn.functional as F

class DownUpBlock(nn.Module):
    def __init__(self, ni, nf, size, stride=1, drop_p=0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(nf)
        self.conv1 = nn.Conv2d(ni, nf, size, stride)
        self.deconv1 = nn.ConvTranspose2d(nf, nf, size, stride)
        self.drop = nn.Dropout(drop_p, inplace=True) if drop_p else None

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.deconv1(x), inplace=True)
        if self.drop: x = self.drop(x)
        x = self.bn(x)
        return x

class ScaleParallelUnit(nn.Module):
    def __init__(self, ni, nf, sizes, stride, drop_p=0.0):
        super().__init__()
        self.conv_layers = nn.ModuleList([DownUpBlock(ni, nf, sizes[i], stride, drop_p) for i in range(len(sizes))])

    def forward(self, x):
        #stream_tmp = []
        #streams = [(idx, torch.cuda.Stream()) for idx, cnn in enumerate(self.conv_layers)]
        #for idx, s in streams:
        #    with torch.cuda.stream(s):
        #        cnn = self.conv_layers[idx]     #<--- how to ensure idx is in sync with the idx in for loop?
        #        stream_tmp.append((idx, cnn(x)))
        #torch.cuda.synchronize() # added synchronize
        #stream_tmp = [t for idx, t in sorted(stream_tmp, key=lambda x:x[0])]
        #stream_tmp.append(x)
        #seq_representation_stream = torch.cat(stream_tmp, dim=1)
        #return seq_representation_stream
        out = [layer(x) for layer in self.conv_layers]
        out.append(x)
        return torch.cat(out, dim=1)

class ScaleParallelBlock(nn.Module):
    def __init__(self, ni=1, nf=40, sizes=[4, 12], num_units=1, stride=1, layer_drop=0.0):
        super().__init__()
        self.parallel_units = nn.ModuleList()
        curr_ni = ni
        for i in range(num_units):
            self.parallel_units.append(ScaleParallelUnit(curr_ni, nf, sizes, stride, layer_drop))
            curr_ni = nf*len(sizes) + curr_ni
        self.full_block = nn.Sequential(self.parallel_units)
        self.output_size = curr_ni

    def forward(self, x):
        return self.full_block(x)

class ScaleParallelClassifier(nn.Module):
    def __init__(self, ni=1, nf=40, sizes=[4, 12], in_height=28, num_units=1, in_width=28, out_size=10, stride=1, nf_1x1=20, layer_drop=0.0, linear_drop=0.0):
        super().__init__()
        self.scale_block = ScaleParallelBlock(ni, nf, sizes, num_units, stride, layer_drop)
        self.conv1x1 = nn.Conv2d(self.scale_block.output_size, nf_1x1, 1, 1)
        self.drop = nn.Dropout(linear_drop, inplace=True) if linear_drop else None
        self.feature_size = nf_1x1*in_width*in_height
        self.linear = nn.Linear(self.feature_size, out_size)

    def forward(self, x):
        x = self.scale_block(x)
        x = self.conv1x1(x)
        x = F.relu(x, inplace=True)
        x = x.view(-1, self.feature_size)
        if self.drop: x = self.drop(x)
        x = F.log_softmax(self.linear(x), dim=1)
        return x
