import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):

    def __init__(self):
        super(PPONetwork, self).__init__()
        self.name = "PPONetwork"

    def save(self, path):
        out_f = os.path.join(path, self.name + ".model")
        torch.save(self.state_dict(), out_f)

    def load(self, path):
        in_f = os.path.join(path, self.name + ".model")
        self.load_state_dict(torch.load(in_f))


class LinearNN(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(LinearNN, self).__init__()
        self.name = name
        self.need_softmax = need_softmax

        self.l1 = nn.Linear(in_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out_dim)


    def forward(self, _input):

        out = self.l1(_input)
        out = F.relu(out)

        out = self.l2(out)
        out = F.relu(out)

        out = self.l3(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out



class LinearNN2(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False):

        super(LinearNN2, self).__init__()
        self.name = name
        self.need_softmax = need_softmax

        self.l1 = nn.Linear(in_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, out_dim)

        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, _input):

        out = self.l1(_input)
        out = self.l_relu(out)

        out = self.l2(out)
        out = self.l_relu(out)

        out = self.l3(out)
        out = self.l_relu(out)

        out = self.l4(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out
