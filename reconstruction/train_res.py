
from __future__ import print_function

from math import log10

import torch
import argparse
import torch.nn as nn
import os.path as osp
import torch.utils.data
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from loader import TransientObjectLoader

from super_res import Net


parser = argparse.ArgumentParser(description='Transient object super-res '
                                             'reconstruction')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='S',
                    help='Learning rate')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--data', type=str, default='../../new_stamps',
                    help='Path to the folder that contains the images')
parser.add_argument('--save', type=str, default='model_res.pt',
                    help='path to save the final model')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    TransientObjectLoader(args.data, train=True,
                          transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    TransientObjectLoader(args.data, train=False,
                          transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


load_ext = False
model = Net(1)
if osp.exists(args.save):
    with open(args.save, 'rb') as f:
        state_dict = torch.load(f)
        # discard = [x for x in state_dict if x.startswith('fc1')]
        # state = model.state_dict()
        model.load_state_dict(state_dict)
    load_ext = True

if args.cuda:
    model.cuda()

reconstruction_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        # print(data.size())
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon = model(data)
        # recon_batch, mu, logvar = model(data)
        # loss = loss_function(recon_batch, data, mu, logvar)
        loss = reconstruction_function(recon, data)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('\n====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    avg_psnr = 0
    for data in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon = model(data)
        loss = reconstruction_function(recon, data)
        test_loss += loss.data[0]
        psnr = 10 * log10(1 / loss.data[0])
        avg_psnr += psnr

    test_loss /= len(test_loader.dataset)
    avg_psnr /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} | Avg PSNR: {:.4f}\n'.format(
        test_loss, avg_psnr))
    return test_loss


if __name__ == '__main__':
    if not load_ext:
        best_test_loss = None
    else:
        best_test_loss = test(0)

    try:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test_loss = test(epoch)

            if not best_test_loss or test_loss < best_test_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_test_loss = test_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        # model = Net()
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    test(epoch)
