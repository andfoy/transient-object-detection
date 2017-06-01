
from __future__ import print_function

import torch
import argparse
import torch.nn as nn
import os.path as osp
import torch.utils.data
import torch.optim as optim
from segnet import make_segnet
from torchvision import transforms
from torch.autograd import Variable
from loader import TransientObjectLoader


parser = argparse.ArgumentParser(description='Transient object Segnet '
                                             'AE reduction')
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
parser.add_argument('--data', type=str,
                    default='../new_stamps',
                    help='Path to the folder that contains the images')
parser.add_argument('--save', type=str, default='model_segnet.pt',
                    help='path to save the final model')

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


model = make_segnet()

if osp.exists(args.save):
    print("Loading snapshot...")
    with open(args.save, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
else:
    vgg_state = torch.load('vgg16_bn-6c64b313.pth')
    state_dict = model.state_dict()
    vgg_layers = [k for k in vgg_state if k.startswith('features')]
    for layer in vgg_layers:
        # if layer.startswith('features'):
        state_dict[layer] = vgg_state[layer]

    vgg_layers = vgg_layers[1:][::-1]
    deconv_layers = [k for k in state_dict if k.startswith('deconv')]
    for layer, vgg in zip(deconv_layers, vgg_layers):
        state_dict[layer] = vgg_state[vgg]

    model.load_state_dict(state_dict)
