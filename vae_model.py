
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, cuda=True):
        super(VAE, self).__init__()

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 800)
        self.fc12 = nn.Linear(800, 500)
        self.fc21 = nn.Linear(500, 100)
        self.fc22 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 500)
        self.fc41 = nn.Linear(500, 800)
        self.fc4 = nn.Linear(800, 1024)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.cuda = cuda

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.fc12(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h3 = self.relu(self.fc41(h3))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1024))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
