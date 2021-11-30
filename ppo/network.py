import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LinearNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinearNN, self).__init__()

        self.l1 = nn.Linear(in_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out_dim)


    def forward(self, _input):

        out = self.l1(_input)
        out = F.relu(out)

        out = self.l2(out)
        out = F.relu(out)

        out = self.l3(out)

        return out
