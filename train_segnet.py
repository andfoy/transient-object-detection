
from __future__ import print_function

import torch
import argparse
import torch.nn as nn
import os.path as osp
import torch.utils.data
import torch.optim as optim
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
                    help='Path to the folder that contains the images')
parser.add_argument('--save', type=str, default='model_vae.pt',
                    help='path to save the final model')
