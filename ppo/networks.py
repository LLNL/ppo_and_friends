import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):

    def __init__(self, **kwargs):
        super(PPONetwork, self).__init__()
        self.name = "PPONetwork"
        self.uses_batch_norm = False

    def save(self, path):
        out_f = os.path.join(path, self.name + ".model")
        torch.save(self.state_dict(), out_f)

    def load(self, path):
        in_f = os.path.join(path, self.name + ".model")
        self.load_state_dict(torch.load(in_f))


class PPOConv2dNetwork(PPONetwork):

    def __init__(self, **kwargs):
        super(PPOConv2dNetwork, self).__init__(**kwargs)
        self.name = "PPOConv2dNetwork"


class SplitObservationNetwork(PPONetwork):

    def __init__(self, split_start, **kwargs):
        super(SplitObservationNetwork, self).__init__(**kwargs)

        if split_start <= 0:
            msg  = "ERROR: SplitObservationNetwork requires a split start "
            msg += "> 0."
            print(msg)
            sys.exit(1)

        self.name = "PPOSplitObservationNetwork"
        self.split_start = split_start


def get_conv2d_out_size(in_size,
                        padding,
                        kernel_size,
                        stride):
        out_size = int(((in_size + 2.0 * padding - (kernel_size - 1) - 1)\
            / stride) + 1)
        return out_size

def get_maxpool2d_out_size(in_size,
                           padding,
                           kernel_size,
                           stride):
    return get_conv2d_out_size(in_size, padding, kernel_size, stride)


########################################################################
#                        Observation Encoders                          #
########################################################################

class LinearObservationEncoder(nn.Module):

    def __init__(self,
                 obs_dim,
                 encoded_dim,
                 hidden_size,
                 **kwargs):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses a simple feed-forward network.
        """

        super(LinearObservationEncoder, self).__init__(**kwargs)

        self.l_relu = nn.LeakyReLU()

        self.enc_1  = nn.Linear(obs_dim, hidden_size)
        self.enc_2  = nn.Linear(hidden_size, hidden_size)
        self.enc_3  = nn.Linear(hidden_size, hidden_size)
        self.enc_4  = nn.Linear(hidden_size, encoded_dim)

    def forward(self,
                obs):

        obs = obs.flatten(start_dim = 1)

        enc_obs = self.enc_1(obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_2(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_3(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.enc_4(enc_obs)
        return enc_obs


class Conv2dObservationEncoder(nn.Module):

    def __init__(self,
                 in_shape,
                 encoded_dim,
                 **kwargs):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses 2d convolutions followed by
            linear layers.
        """

        super(Conv2dObservationEncoder, self).__init__(**kwargs)

        self.l_relu = nn.LeakyReLU()

        channels   = in_shape[0]
        height     = in_shape[1]
        width      = in_shape[2]

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_1 = nn.Conv2d(channels, 8, kernel_size=5, stride=1)
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.mp_1 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_2 = nn.Conv2d(16, 16, kernel_size=5, stride=1)
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.mp_2 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_3 = nn.Conv2d(16, 16, kernel_size=5, stride=1)
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 0
        strd = 1
        self.mp_3 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        self.linear_encoder = LinearObservationEncoder(
            height * width * 16,
            encoded_dim,
            encoded_dim)


    def forward(self,
                obs):

        enc_obs = self.conv_1(obs)
        enc_obs = self.mp_1(enc_obs)

        enc_obs = self.conv_2(enc_obs)
        enc_obs = self.mp_2(enc_obs)

        enc_obs = self.conv_3(enc_obs)
        enc_obs = self.mp_3(enc_obs)

        enc_obs = enc_obs.flatten(start_dim = 1)

        enc_obs = self.linear_encoder(enc_obs)

        return enc_obs


class Conv2dObservationEncoder_orig(nn.Module):

    def __init__(self,
                 in_shape,
                 encoded_dim,
                 hidden_size,
                 **kwargs):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses 2d convolutions followed by
            linear layers.
        """

        super(Conv2dObservationEncoder_orig, self).__init__(**kwargs)

        self.l_relu = nn.LeakyReLU()

        channels   = in_shape[0]
        height     = in_shape[1]
        width      = in_shape[2]

        k_s  = 5
        pad  = 0
        strd = 1
        self.conv_1 = nn.Conv2d(channels, 32, kernel_size=5, stride=1)
        height      = get_conv2d_out_size(height, pad, k_s, strd)
        width       = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 5
        pad  = 0
        strd = 2
        self.mp_1 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 5
        pad  = 0
        strd = 1
        self.conv_2   = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        height       = get_conv2d_out_size(height, pad, k_s, strd)
        width        = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 5
        pad  = 0
        strd = 2
        self.mp_2 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 5
        pad  = 0
        strd = 1
        self.conv_3   = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        height       = get_conv2d_out_size(height, pad, k_s, strd)
        width        = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 5
        pad  = 0
        strd = 2
        self.mp_3 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height    = get_maxpool2d_out_size(height, pad, k_s, strd)
        width     = get_maxpool2d_out_size(width, pad, k_s, strd)

        self.l1  = nn.Linear(height * width * 32, hidden_size)
        self.l2  = nn.Linear(hidden_size, hidden_size)
        self.l3  = nn.Linear(hidden_size, hidden_size)
        self.l4  = nn.Linear(hidden_size, encoded_dim)


    def forward(self,
                obs):

        enc_obs = self.conv_1(obs)
        enc_obs = self.mp_1(enc_obs)

        enc_obs = self.conv_2(enc_obs)
        enc_obs = self.mp_2(enc_obs)

        enc_obs = self.conv_3(enc_obs)
        enc_obs = self.mp_3(enc_obs)

        enc_obs = enc_obs.flatten(start_dim = 1)

        enc_obs = self.l1(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.l2(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.l3(enc_obs)
        enc_obs = self.l_relu(enc_obs)

        enc_obs = self.l4(enc_obs)

        return enc_obs



########################################################################
#                        Actor Critic Networks                         #
########################################################################


class SimpleFeedForward(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False,
                 activation   = torch.nn.ReLU(),
                 hidden_size  = 128,
                 **kwargs):

        super(SimpleFeedForward, self).__init__(**kwargs)

        self.name         = name
        self.need_softmax = need_softmax
        self.activation   = activation

        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, out_dim)

    def forward(self, _input):
        out = _input.flatten(start_dim = 1)

        out = self.l1(out)
        out = self.activation(out)

        out = self.l2(out)
        out = self.activation(out)

        out = self.l3(out)
        out = self.activation(out)

        out = self.l4(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


class SimpleSplitObsNetwork(SplitObservationNetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False,
                 hidden_size  = 64,
                 **kwargs):

        super(SimpleSplitObsNetwork, self).__init__(**kwargs)

        self.name         = name
        self.need_softmax = need_softmax
        self.activation   = nn.ReLU()

        side_1_dim = self.split_start
        side_2_dim = in_dim - self.split_start

        # TODO: in the orignal paper, there is a "low level" section and
        # a "high level" section. The low level section handles
        # proprioceptive information (joints, positions, etc.), and the
        # high level section sees everything but only sends a signal every
        # K iterations. A later paper uses a similar technique, but it's
        # unclear if they keep the "K iteration" approach (they're very
        # vague). At any rate, this later approach is slighty different in
        # that it splits between proprioceptive and exteroceptive (sensory
        # information about the environment). That's basically what we're
        # doing here, except we are using the same architecture for both
        # parts of the observation. Both of the previous papers use different
        # architectures for the different sections.
        self.s1_net = SimpleFeedForward(
            name       = self.name + "_s1",
            in_dim     = side_1_dim,
            out_dim    = hidden_size,
            activation = self.activation)

        self.s2_net = SimpleFeedForward(
            name       = self.name + "_s2",
            in_dim     = side_2_dim,
            out_dim    = hidden_size,
            activation = self.activation)

        inner_hidden_size  = hidden_size * 2

        self.full_l1 = nn.Linear(inner_hidden_size, inner_hidden_size)
        self.full_l2 = nn.Linear(inner_hidden_size, inner_hidden_size)
        self.full_l3 = nn.Linear(inner_hidden_size, out_dim)

    def forward(self, _input):
        out = _input.flatten(start_dim = 1)

        s1_out = out[:, 0 : self.split_start]
        s2_out = out[:, self.split_start : ]

        #
        # Side 1.
        #
        s1_out = self.s1_net(s1_out)

        #
        # Side 2.
        #
        s2_out = self.s2_net(s2_out)

        #
        # Full layers.
        #
        out = torch.cat((s1_out, s2_out), dim=1)

        out = self.full_l1(out)
        out = self.activation(out)

        out = self.full_l2(out)
        out = self.activation(out)

        out = self.full_l3(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


class AtariRAMNetwork(PPONetwork):

    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 need_softmax = False,
                 **kwargs):

        super(AtariRAMNetwork, self).__init__(**kwargs)
        self.name            = name
        self.need_softmax    = need_softmax
        self.uses_batch_norm = True

        self.a_f = torch.nn.ReLU()

        self.l1 = nn.Linear(in_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.l2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.l3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.l4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.l5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.l6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)

        self.l7 = nn.Linear(32, 16)
        self.bn7 = nn.BatchNorm1d(16)

        self.l8 = nn.Linear(16, 8)
        self.bn8 = nn.BatchNorm1d(8)

        self.l9 = nn.Linear(8, out_dim)


    def forward(self, _input):

        out = self.l1(_input)
        out = self.bn1(out)
        out = self.a_f(out)

        out = self.l2(out)
        out = self.bn2(out)
        out = self.a_f(out)

        out = self.l3(out)
        out = self.bn3(out)
        out = self.a_f(out)

        out = self.l4(out)
        out = self.bn4(out)
        out = self.a_f(out)

        out = self.l5(out)
        out = self.bn5(out)
        out = self.a_f(out)

        out = self.l6(out)
        out = self.bn6(out)
        out = self.a_f(out)

        out = self.l7(out)
        out = self.bn7(out)
        out = self.a_f(out)

        out = self.l8(out)
        out = self.bn8(out)
        out = self.a_f(out)

        out = self.l9(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


class AtariPixelNetwork(PPOConv2dNetwork):

    def __init__(self,
                 name,
                 in_shape,
                 out_dim,
                 need_softmax = False,
                 **kwargs):

        super(AtariPixelNetwork, self).__init__(**kwargs)

        self.name         = name
        self.need_softmax = need_softmax
        self.a_f          = torch.nn.ReLU()

        channels   = in_shape[0]
        height     = in_shape[1]
        width      = in_shape[2]

        k_s  = 8
        strd = 4
        pad  = 0
        self.conv1 = nn.Conv2d(channels, 16,
            kernel_size=k_s, stride=strd, padding=pad)
        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 1
        strd = 1
        self.mp1 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height = get_maxpool2d_out_size(height, pad, k_s, strd)
        width  = get_maxpool2d_out_size(width, pad, k_s, strd)

        k_s  = 4
        strd = 2
        pad  = 0
        self.conv2 = nn.Conv2d(16, 32,
            kernel_size=k_s, stride=strd, padding=pad)
        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        pad  = 1
        strd = 1
        self.mp2 = nn.MaxPool2d(kernel_size=k_s, padding=pad, stride=strd)
        height = get_maxpool2d_out_size(height, pad, k_s, strd)
        width  = get_maxpool2d_out_size(width, pad, k_s, strd)

        self.l1 = nn.Linear(height * width * 32, 256)
        self.l2 = nn.Linear(256, out_dim)


    def forward(self, _input):
        out = self.conv1(_input)
        out = self.mp1(out)
        out = self.a_f(out)

        out = self.conv2(out)
        out = self.mp2(out)
        out = self.a_f(out)

        out = out.flatten(start_dim=1)

        out = self.l1(out)
        out = self.a_f(out)

        out = self.l2(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


########################################################################
#                           ICM Networks                               #
########################################################################


class LinearInverseModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_size,
                 **kwargs):
        """
            An implementation of the inverse model for the ICM method. The
            inverse model learns to predict the action that was performed
            when given the current and next states. States may also be
            observations, and the observations are typically encoded into
            a form that only contains information that the actor needs to
            make choices.

            This implementation uses a simple feed-forward network.
        """

        super(LinearInverseModel, self).__init__(**kwargs)

        self.l_relu = nn.LeakyReLU()

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_1 = nn.Linear(in_dim, hidden_size)
        self.inv_2 = nn.Linear(hidden_size, hidden_size)
        self.inv_3 = nn.Linear(hidden_size, out_dim)

    def forward(self,
                enc_obs_1,
                enc_obs_2):

        obs_input = torch.cat((enc_obs_1, enc_obs_2), dim=1)

        out = self.inv_1(obs_input)
        out = self.l_relu(out)

        out = self.inv_2(out)
        out = self.l_relu(out)

        out = self.inv_3(out)
        out = F.softmax(out, dim=-1)

        return out


class LinearForwardModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 act_dim,
                 hidden_size,
                 action_type,
                 **kwargs):
        """
            The forward model of the ICM method. In short, this learns to
            predict the changes in state given a current state and action.
            In reality, our states can be observations, and our observations
            are typically encoded into a form that only contains information
            that an actor needs to make decisions.

            This implementation uses a simple feed-forward network.
        """

        super(LinearForwardModel, self).__init__(**kwargs)

        self.l_relu      = nn.LeakyReLU()
        self.action_type = action_type
        self.act_dim     = act_dim

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.f_1 = nn.Linear(in_dim, hidden_size)
        self.f_2 = nn.Linear(hidden_size, hidden_size)
        self.f_3 = nn.Linear(hidden_size, out_dim)

    def forward(self,
                enc_obs_1,
                actions):

        actions = actions.flatten(start_dim = 1)

        if self.action_type == "discrete":
            actions = torch.nn.functional.one_hot(actions,
                num_classes=self.act_dim).float()

            #
            # One-hot adds an extra dimension. Let's get rid of that bit.
            #
            actions = actions.squeeze(-2)

        _input = torch.cat((enc_obs_1, actions), dim=1)

        #
        # Predict obs_2 given obs_1 and actions.
        #
        out = self.f_1(_input)
        out = self.l_relu(out)

        out = self.f_2(out)
        out = self.l_relu(out)

        out = self.f_3(out)
        return out


class ICM(PPONetwork):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_type,
                 obs_encoder  = LinearObservationEncoder,
                 reward_scale = 0.01,
                 **kwargs):
        """
            The Intrinsic Curiosit Model (ICM).
        """

        super(ICM, self).__init__(**kwargs)

        self.act_dim      = act_dim
        self.reward_scale = reward_scale
        self.action_type  = action_type

        self.l_relu       = nn.LeakyReLU()
        self.ce_loss      = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss     = nn.MSELoss(reduction="none")

        encoded_obs_dim   = 128
        hidden_dims       = 128

        #
        # Observation encoder.
        #
        self.obs_encoder = obs_encoder(
            obs_dim,
            encoded_obs_dim,
            hidden_dims)

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_model = LinearInverseModel(
            encoded_obs_dim * 2, 
            act_dim,
            hidden_dims)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.forward_model = LinearForwardModel(
            encoded_obs_dim + act_dim,
            encoded_obs_dim,
            act_dim,
            hidden_dims,
            action_type)

    def forward(self,
                obs_1,
                obs_2,
                actions):

        #
        # First, encode the observations.
        #
        enc_obs_1 = self.obs_encoder(obs_1)
        enc_obs_2 = self.obs_encoder(obs_2)

        #
        # Inverse model prediction.
        #
        action_pred = self.inv_model(enc_obs_1, enc_obs_2)

        if self.action_type == "discrete":
            inv_loss = self.ce_loss(action_pred, actions.flatten())
        elif self.action_type == "continuous":
            inv_loss = self.mse_loss(action_pred, actions).mean()

        #
        # Forward model prediction.
        #
        obs_2_pred = self.forward_model(enc_obs_1, actions)
        f_loss = self.mse_loss(obs_2_pred, enc_obs_2)

        intrinsic_reward = (self.reward_scale / 2.0) * f_loss.sum(dim=-1)
        f_loss           = 0.5 * f_loss.mean()

        return intrinsic_reward, inv_loss, f_loss

