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

    def __init__(self,
                 name,
                 in_length):

        super(LinearDQN, self).__init__()
        self.name = name

        self.fc1 = nn.Linear(in_features  = in_length, out_features = 24)
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


class LinearDQN3(DQN):

    def __init__(self,
                 name,
                 in_length):

        super(LinearDQN3, self).__init__()
        self.name = name

        self.fc1 = nn.Linear(in_features  = in_length, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 128)
        self.fc3 = nn.Linear(in_features = 128, out_features = 128)
        self.fc4 = nn.Linear(in_features = 128, out_features = 2)

    def forward(self, _input):

        output = _input.flatten(start_dim = 1)

        output = self.fc1(output)
        output = t_functional.relu(output)

        output = self.fc2(output)
        output = t_functional.relu(output)

        output = self.fc3(output)
        output = t_functional.relu(output)

        output = self.fc4(output)

        return output

class SimpleDQN(DQN):

    def __init__(self,
                 name,
                 in_length):

        super(SimpleDQN, self).__init__()
        self.name = name

        self.fc1   = nn.Linear(in_features = in_length, out_features = 32)
        self.fc2   = nn.Linear(in_features = 32, out_features = 2)

    def forward(self, _input):

        output = _input.flatten(start_dim = 1)

        output = self.fc1(output)
        output = torch.relu(output)

        output = self.fc2(output)

        return output


class Linear2DQN(DQN):

    def __init__(self,
                 name,
                 in_length):

        super(Linear2DQN, self).__init__()
        self.name = name

        self.fc1 = nn.Linear(in_features  = in_length, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 256)
        self.fc3 = nn.Linear(in_features = 256, out_features = 128)
        self.fc4 = nn.Linear(in_features = 128, out_features = 2)

    def forward(self, _input):

        output = _input.flatten(start_dim = 1)

        output = self.fc1(output)
        output = t_functional.relu(output)

        output = self.fc2(output)
        output = t_functional.relu(output)

        output = self.fc3(output)
        output = t_functional.relu(output)

        output = self.fc4(output)

        return output


class Conv1dDQN(DQN):

    def __init__(self,
                 name,
                 in_length):

        super(Conv1dDQN, self).__init__()
        self.name = name

        kernel_size = in_length - 1
        padding = int((kernel_size - 1) / 2)

        self.conv1 = nn.Conv1d(
            1,
            36,
            kernel_size = kernel_size,
            padding     = padding)

        self.conv2 = nn.Conv1d(
            36,
            36,
            kernel_size = kernel_size,
            padding     = padding)

        self.conv3 = nn.Conv1d(
            36,
            256,
            kernel_size = kernel_size,
            padding     = padding)

        self.conv4 = nn.Conv1d(
            36,
            36,
            kernel_size = kernel_size,
            padding     = padding)

        self.fc1 = nn.Linear(in_features = in_length * 36, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 2)

    def forward(self, _input):

        output = _input.flatten(start_dim = 1).unsqueeze(1)

        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)

        output = output.flatten(start_dim = 1)

        output = self.fc1(output)
        output = torch.relu(output)

        output = self.fc2(output)

        return output


class ConvDQN(DQN):

    def __init__(self,
                 name,
                 img_height,
                 img_width):
        super(ConvDQN, self).__init__()
        self.name = name

        padding = 0
        get_new_size = lambda in_s, k_s, s_s: int(((in_s + (2 * padding) - (k_s - 1) - 1) / s_s) + 1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5,
            stride=3, padding=0)

        out_height = get_new_size(img_height, 5, 3)
        out_width  = get_new_size(img_width, 5, 3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4,
            stride=2, padding=0)

        out_height = get_new_size(out_height, 4, 2)
        out_width  = get_new_size(out_width, 4, 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,
            stride=1, padding=0)

        out_height = get_new_size(out_height, 3, 1)
        out_width  = get_new_size(out_width, 3, 1)

        in_length  = out_height * out_width * 64

        self.fc1 = nn.Linear(in_features = in_length, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 256)
        self.fc3 = nn.Linear(in_features = 256, out_features = 64)
        self.fc4 = nn.Linear(in_features = 64, out_features = 2)


    def forward(self, _input):

        output = self.conv1(_input)
        output = torch.relu(output)

        output = self.conv2(output)
        output = torch.relu(output)

        output = self.conv3(output)
        output = torch.relu(output)

        output = output.flatten(start_dim = 1)

        output = self.fc1(output)
        output = t_functional.relu(output)

        output = self.fc2(output)
        output = t_functional.relu(output)

        output = self.fc3(output)
        output = t_functional.relu(output)

        output = self.fc4(output)

        return output


class ConvDQN2(DQN):

    def __init__(self,
                 name,
                 img_height,
                 img_width):
        super(ConvDQN2, self).__init__()
        self.name = name

        padding = 0
        get_new_size = lambda in_s, k_s, s_s: int(((in_s + (2 * padding) - (k_s - 1) - 1) / s_s) + 1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5,
            stride=1, padding=0)

        out_height = get_new_size(img_height, 5, 1)
        out_width  = get_new_size(img_width, 5, 1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5,
            stride=1, padding=0)

        out_height = get_new_size(out_height, 5, 1)
        out_width  = get_new_size(out_width, 5, 1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5,
            stride=1, padding=0)

        out_height = get_new_size(out_height, 5, 1)
        out_width  = get_new_size(out_width, 5, 1)

        in_length  = out_height * out_width * 64

        self.fc1 = nn.Linear(in_features = in_length, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 256)
        self.fc3 = nn.Linear(in_features = 256, out_features = 64)
        self.fc4 = nn.Linear(in_features = 64, out_features = 2)


    def forward(self, _input):

        output = self.conv1(_input)
        output = torch.relu(output)

        output = self.conv2(output)
        output = torch.relu(output)

        output = self.conv3(output)
        output = torch.relu(output)

        output = output.flatten(start_dim = 1)

        output = self.fc1(output)
        output = t_functional.relu(output)

        output = self.fc2(output)
        output = t_functional.relu(output)

        output = self.fc3(output)
        output = t_functional.relu(output)

        output = self.fc4(output)

        return output
