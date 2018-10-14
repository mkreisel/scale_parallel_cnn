from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        self.parallel_units = []
        curr_ni = ni
        for i in range(num_units):
            self.parallel_units.append(ScaleParallelUnit(curr_ni, nf, sizes, stride, layer_drop))
            curr_ni = nf*len(sizes) + curr_ni
        self.full_block = nn.Sequential(*self.parallel_units)
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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = ScaleParallelClassifier(num_units=2, nf=20, nf_1x1=20, sizes=[4,12]).to(device)
    print(model.drop == None)
    print("Total trainable params: %i" % count_parameters(model))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    print("Dropping LR")
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / 10
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
