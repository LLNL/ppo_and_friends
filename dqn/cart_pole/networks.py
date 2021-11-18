import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as t_functional

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.name = "DQN"

    def save(self, path):
        out_f = os.path.join(path, self.name + ".model")
        torch.save(self.state_dict(), out_f)

    def load(self, path):
        in_f = os.path.join(path, self.name + ".model")
        self.load_state_dict(torch.load(in_f))


class LinearDQN(DQN):

    def __init__(self, img_height, img_width):
        super(LinearDQN, self).__init__()
        self.name = "LinearDQN"

        self.fc1 = nn.Linear(
            in_features  = img_height * img_width * 3,
            out_features = 24)
        self.fc2 = nn.Linear(in_features = 24, out_features = 32)
        self.fc3 = nn.Linear(in_features = 32, out_features = 2)

    def forward(self, _input):

        output = _input.flatten(start_dim = 1)

        output = self.fc1(output)
        output = t_functional.relu(output)

        output = self.fc2(output)
        output = t_functional.relu(output)

        output = self.fc3(output)

        return output


class ConvDQN(DQN):

    def __init__(self, img_height, img_width):
        super(ConvDQN, self).__init__()
        self.name = "ConvDQN"

        kernel_size = 4
        padding     = 1

        get_new_size = lambda in_s : int(in_s + (2 * padding) - (kernel_size - 1))

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size,
            stride=1, padding=padding)
        self.bn1   = nn.BatchNorm2d(16)

        out_height = get_new_size(img_height)
        out_width  = get_new_size(img_width)

        #self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size,
        #    stride=1, padding=padding)
        #self.bn2   = nn.BatchNorm2d(32)

        #out_height = get_new_size(out_height)
        #out_width  = get_new_size(out_width)

        #self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size,
        #    stride=1, padding=padding)
        #self.bn3   = nn.BatchNorm2d(32)

        #out_height = get_new_size(out_height)
        #out_width  = get_new_size(out_width)

        self.fc1 = nn.Linear(
            in_features  = out_height * out_width * 16,
            out_features = 24)
        self.fc2 = nn.Linear(in_features = 24, out_features = 32)
        self.fc3 = nn.Linear(in_features = 32, out_features = 2)

    def forward(self, _input):

        output = self.conv1(_input)
        output = self.bn1(output)

        #output = self.conv2(output)
        #output = self.bn2(output)

        #output = self.conv3(output)
        #output = self.bn3(output)

        output = output.flatten(start_dim = 1)

        output = self.fc1(output)
        output = t_functional.relu(output)

        output = self.fc2(output)
        output = t_functional.relu(output)

        output = self.fc3(output)

        return output
