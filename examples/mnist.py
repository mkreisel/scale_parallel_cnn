from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_large1 = nn.Conv2d(1, 10, kernel_size=8, stride=4)
        self.conv_medium1 = nn.Conv2d(1, 10, kernel_size=4, stride=2)
        self.conv_small1 = nn.Conv2d(1, 10, kernel_size=2)
        self.conv_large_1x1 = nn.Conv2d(10, 10, kernel_size=1)
        self.conv_medium_1x1 = nn.Conv2d(10, 10, kernel_size=1)
        self.conv_small_1x1 = nn.Conv2d(10, 10, kernel_size=1)
        self.conv_large_1x1_2 = nn.Conv2d(4, 4, kernel_size=1)
        self.conv_medium_1x1_2 = nn.Conv2d(4, 4, kernel_size=1)
        self.conv_small_1x1_2 = nn.Conv2d(4, 4, kernel_size=1)
        self.kernel_generator = nn.Linear(270, 336)
        self.fc1 = nn.Linear(378, 10)
        self.upper_head =nn.Linear(270, 10)
        self.lower_head =nn.Linear(108, 10)

    def forward(self, x):
        c1 = F.relu(self.conv_large1(x))
        c2 = F.relu(self.conv_medium1(x))
        c3 = F.relu(self.conv_small1(x))
        c1 = F.relu(F.max_pool2d(self.conv_large_1x1(c1), 2))
        c2 = F.relu(F.max_pool2d(self.conv_medium_1x1(c2), 4))
        c3 = F.relu(F.max_pool2d(self.conv_small_1x1(c3), 8))
        m1 = torch.cat([c1, c2, c3], dim=1).view(-1, 270)
        kernels = self.kernel_generator(m1)
        large_kernels = kernels[-1,:256].view(4, 1, 8, 8)
        medium_kernels = kernels[-1, 256:320].view(4, 1, 4, 4)
        small_kernels = kernels[-1,320:].view(4, 1, 2, 2)
        c1_prime = torch.nn.functional.conv2d(x, large_kernels, stride=4)
        c2_prime = torch.nn.functional.conv2d(x, medium_kernels, stride=2)
        c3_prime = torch.nn.functional.conv2d(x, small_kernels)
        c1_prime = F.relu(F.max_pool2d(self.conv_large_1x1_2(c1_prime), 2))
        c2_prime = F.relu(F.max_pool2d(self.conv_medium_1x1_2(c2_prime), 4))
        c3_prime = F.relu(F.max_pool2d(self.conv_small_1x1_2(c3_prime), 8))
        m2 = torch.cat([c1_prime, c2_prime, c3_prime], dim=1).view(-1, 108)
        return F.log_softmax(self.fc1(torch.cat([m1,m2], dim=1)), dim=1), F.log_softmax(self.upper_head(m1), dim=1), F.log_softmax(self.lower_head(m2), dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, upper_out, lower_out = model(data)
        loss = F.nll_loss(output, target) + F.nll_loss(upper_out, target) + F.nll_loss(lower_out, target)
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
            output, upper_out, lower_out = model(data)
            test_loss += F.nll_loss(lower_out, target, reduction='sum').item() + F.nll_loss(upper_out, target, reduction='sum').item() + F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
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


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
