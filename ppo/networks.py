import torch
import os
from torch import nn
import math
from numbers import Real
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import numpy as np
import sys

class PPONetwork(nn.Module):

    def __init__(self,
                 name,
                 **kw_args):

        super(PPONetwork, self).__init__()
        self.name = name

    def save(self, path):
        out_f = os.path.join(path, self.name + ".model")
        torch.save(self.state_dict(), out_f)

    def load(self, path):
        in_f = os.path.join(path, self.name + ".model")
        self.load_state_dict(torch.load(in_f))


class PPOActorCriticNetwork(PPONetwork):

    def __init__(self,
                 action_type,
                 out_dim,
                 **kw_args):

        super(PPOActorCriticNetwork, self).__init__(**kw_args)

        self.action_type  = action_type
        self.need_softmax = False

        #
        # Actors have special roles.
        #
        if self.name == "actor":

            if action_type == "discrete":
                self.need_softmax = True
                self.distribution  = CategoricalDistribution(**kw_args)
            elif action_type == "continuous":
                self.distribution = GaussianDistribution(out_dim, **kw_args)


class PPOConv2dNetwork(PPOActorCriticNetwork):

    def __init__(self, **kw_args):
        super(PPOConv2dNetwork, self).__init__(**kw_args)


class SplitObservationNetwork(PPOActorCriticNetwork):

    def __init__(self, split_start, **kw_args):
        super(SplitObservationNetwork, self).__init__(**kw_args)

        if split_start <= 0:
            msg  = "ERROR: SplitObservationNetwork requires a split start "
            msg += "> 0."
            print(msg)
            sys.exit(1)

        self.split_start = split_start


class CategoricalDistribution(object):

    def __init__(self,
                 **kw_args):
        pass

    def get_distribution(self, probs):
        return Categorical(probs)

    def get_log_probs(self, dist, actions):
        return dist.log_prob(actions)

    def sample_distribution(self, dist):
        sample = dist.sample()
        return sample, sample

    def get_entropy(self, dist, _):
        return dist.entropy()


class GaussianDistribution(nn.Module):

    def __init__(self,
                 act_dim,
                 std_offset = 0.5,
                 **kw_args):

        super(GaussianDistribution, self).__init__()

        #
        # arXiv:2006.05990v1 suggests an offset of -0.5 is best for
        # most continuous control tasks, but there are some which perform
        # better with higher values.
        # TODO: We might want to make this adjustable.
        #
        log_std = torch.as_tensor(-std_offset * np.ones(act_dim, dtype=np.float32))
        self.log_std = torch.nn.Parameter(log_std)

    def get_distribution(self, action_mean):

        #
        # arXiv:2006.05990v1 suggests that softplus can perform
        # slightly better than exponentiation.
        # TODO: add option to use softplus or exp.
        #
        std = nn.functional.softplus(self.log_std)
        return Normal(action_mean, std.cpu())

    def get_log_probs(self, dist, pre_tanh_actions, epsilon=1e-6):
        #
        # NOTE: while wrapping our samples in tanh does change
        # our distribution, arXiv:2006.05990v1 suggests that this
        # doesn't affect calculating loss or KL divergence. However,
        # I've found that some environments have some serious issues
        # with just using the log_prob from the distribution.
        # The following calculation is taken from arXiv:1801.01290v2,
        # but I've also added clamps for safety.
        #
        normal_log_probs = dist.log_prob(pre_tanh_actions)
        normal_log_probs = torch.clamp(normal_log_probs, -100, 100)
        normal_log_probs = normal_log_probs.sum(dim=-1)

        tanh_prime = 1.0 - torch.pow(torch.tanh(pre_tanh_actions), 2)
        tanh_prime = torch.clamp(tanh_prime, epsilon, None)
        s_log      = torch.log(tanh_prime).sum(dim=-1)
        return normal_log_probs - s_log

    def sample_distribution(self, dist):
        sample      = dist.sample()
        tanh_sample = torch.tanh(sample)
        return tanh_sample, sample

    def get_entropy(self, dist, pre_tanh_actions, epsilon=1e-6):
        #
        # This is a bit odd here... arXiv:2006.05990v1 suggests using
        # tanh to move the actions into a [-1, 1] range, but this also
        # changes the probability densities. They suggest is okay for most
        # situations because if the differntiation (see above comments),
        # but it does affect the entropy. They suggest using the equation
        # the following equation:
        #    Ex[-log(x) + log(tanh^prime (x))] s.t. x is the pre-tanh
        #    computed probability distribution.
        # Note that this is the same as our log probs when using tanh but
        # negated.
        #
        return -self.get_log_probs(dist, pre_tanh_actions)


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

#TODO: reference sources
def init_layer(layer,
               weight_std = np.sqrt(2),
               bias_const = 0.0):

    torch.nn.init.orthogonal_(layer.weight, weight_std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


########################################################################
#                        Observation Encoders                          #
########################################################################

class LinearObservationEncoder(nn.Module):

    def __init__(self,
                 obs_dim,
                 encoded_dim,
                 out_init,
                 hidden_size,
                 activation = nn.ReLU(),
                 **kw_args):
        """
            A simple encoder for encoding observations into
            forms that only contain information needed by the
            actor. In other words, we want to teach this model
            to get rid of any noise that may exist in the observation.
            By noise, we mean anything that does not pertain to
            the actions being taken.

            This implementation uses a simple feed-forward network.
        """

        super(LinearObservationEncoder, self).__init__()

        self.activation = activation

        self.enc_1 = init_layer(nn.Linear(obs_dim, hidden_size))
        self.enc_2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.enc_3 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.enc_4 = init_layer(nn.Linear(hidden_size, encoded_dim),
            weight_std=out_init)

    def forward(self,
                obs):

        obs = obs.flatten(start_dim = 1)

        enc_obs = self.enc_1(obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.enc_2(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.enc_3(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.enc_4(enc_obs)
        return enc_obs


class Conv2dObservationEncoder(nn.Module):

    def __init__(self,
                 in_shape,
                 encoded_dim,
                 out_init,
                 activation = nn.ReLU(),
                 **kw_args):
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

        super(Conv2dObservationEncoder, self).__init__()

        self.activation = activation

        channels   = in_shape[0]
        height     = in_shape[1]
        width      = in_shape[2]

        k_s  = 3
        pad  = 0
        strd = 1
        self.conv_1 = init_layer(nn.Conv2d(channels, 8,
            kernel_size=5, stride=1))
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
        self.conv_2 = init_layer(nn.Conv2d(16, 16, kernel_size=5, stride=1))
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
        self.conv_3 = init_layer(nn.Conv2d(16, 16, kernel_size=5, stride=1))
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
            out_init,
            encoded_dim)


    def forward(self,
                obs):

        enc_obs = self.conv_1(obs)
        enc_obs = self.mp_1(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.conv_2(enc_obs)
        enc_obs = self.mp_2(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = self.conv_3(enc_obs)
        enc_obs = self.mp_3(enc_obs)
        enc_obs = self.activation(enc_obs)

        enc_obs = enc_obs.flatten(start_dim = 1)

        enc_obs = self.linear_encoder(enc_obs)

        return enc_obs

########################################################################
#                        Actor Critic Networks                         #
########################################################################

class SimpleFeedForward(PPOActorCriticNetwork):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 activation   = nn.ReLU(),
                 hidden_size  = 128,
                 **kw_args):

        super(SimpleFeedForward, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.activation = activation

        self.l1 = init_layer(nn.Linear(in_dim, hidden_size))
        self.l2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.l3 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.l4 = init_layer(nn.Linear(hidden_size, out_dim),
            weight_std=out_init)

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
                 in_dim,
                 out_dim,
                 out_init,
                 hidden_left  = 64,
                 hidden_right = 64,
                 activation   = nn.ReLU(),
                 **kw_args):

        super(SimpleSplitObsNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.activation = activation

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
        # information about the environment).
        #
        s1_kw_args = kw_args.copy()
        s1_kw_args["name"] = self.name + "_s1"

        self.s1_net = SimpleFeedForward(
            in_dim     = side_1_dim,
            out_dim    = hidden_left,
            out_init   = np.sqrt(2),
            activation = self.activation,
            **s1_kw_args)

        s2_kw_args = kw_args.copy()
        s2_kw_args["name"] = self.name + "_s2"

        self.s2_net = SimpleFeedForward(
            in_dim     = side_2_dim,
            out_dim    = hidden_right,
            out_init   = np.sqrt(2),
            activation = self.activation,
            **s2_kw_args)

        inner_hidden_size  = hidden_left + hidden_right

        self.full_l1 = init_layer(nn.Linear(
            inner_hidden_size, inner_hidden_size))

        self.full_l2 = init_layer(nn.Linear(inner_hidden_size,
            inner_hidden_size))

        self.full_l3 = init_layer(nn.Linear(inner_hidden_size,
            out_dim), weight_std=out_init)

    def forward(self, _input):
        out = _input.flatten(start_dim = 1)

        s1_out = out[:, 0 : self.split_start]
        s2_out = out[:, self.split_start : ]

        #
        # Side 1 (left side).
        #
        s1_out = self.s1_net(s1_out)

        #
        # Side 2 (right side).
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


class AtariRAMNetwork(PPOActorCriticNetwork):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 activation = nn.ReLU(),
                 **kw_args):

        super(AtariRAMNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.a_f = activation

        self.l1 = init_layer(nn.Linear(in_dim, 1024))
        self.l2 = init_layer(nn.Linear(1024, 512))
        self.l3 = init_layer(nn.Linear(512, 256))
        self.l4 = init_layer(nn.Linear(256, 128))
        self.l5 = init_layer(nn.Linear(128, 64))
        self.l6 = init_layer(nn.Linear(64, 32))
        self.l7 = init_layer(nn.Linear(32, 16))
        self.l8 = init_layer(nn.Linear(16, 8))
        self.l9 = init_layer(nn.Linear(8, out_dim), weight_std=out_init)


    def forward(self, _input):

        out = self.l1(_input)
        out = self.a_f(out)

        out = self.l2(out)
        out = self.a_f(out)

        out = self.l3(out)
        out = self.a_f(out)

        out = self.l4(out)
        out = self.a_f(out)

        out = self.l5(out)
        out = self.a_f(out)

        out = self.l6(out)
        out = self.a_f(out)

        out = self.l7(out)
        out = self.a_f(out)

        out = self.l8(out)
        out = self.a_f(out)

        out = self.l9(out)

        if self.need_softmax:
            out = F.softmax(out, dim=-1)

        return out


class AtariPixelNetwork(PPOConv2dNetwork):

    def __init__(self,
                 in_shape,
                 out_dim,
                 out_init,
                 activation = nn.ReLU(),
                 **kw_args):

        super(AtariPixelNetwork, self).__init__(
            out_dim = out_dim,
            **kw_args)

        self.a_f = activation

        channels   = in_shape[0]
        height     = in_shape[1]
        width      = in_shape[2]

        k_s  = 8
        strd = 4
        pad  = 0
        self.conv1 = init_layer(nn.Conv2d(channels, 32,
            kernel_size=k_s, stride=strd, padding=pad))

        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 4
        strd = 2
        pad  = 0
        self.conv2 = init_layer(nn.Conv2d(32, 64,
            kernel_size=k_s, stride=strd, padding=pad))

        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        k_s  = 3
        strd = 1
        pad  = 0
        self.conv3 = init_layer(nn.Conv2d(64, 64,
            kernel_size=k_s, stride=strd, padding=pad))

        height = get_conv2d_out_size(height, pad, k_s, strd)
        width  = get_conv2d_out_size(width, pad, k_s, strd)

        self.l1 = init_layer(nn.Linear(height * width * 64, 512))
        self.l2 = init_layer(nn.Linear(512, out_dim), weight_std=out_init)


    def forward(self, _input):
        out = self.conv1(_input)
        out = self.a_f(out)

        out = self.conv2(out)
        out = self.a_f(out)

        out = self.conv3(out)
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
                 out_init,
                 hidden_size,
                 activation = nn.ReLU(),
                 **kw_args):
        """
            An implementation of the inverse model for the ICM method. The
            inverse model learns to predict the action that was performed
            when given the current and next states. States may also be
            observations, and the observations are typically encoded into
            a form that only contains information that the actor needs to
            make choices.

            This implementation uses a simple feed-forward network.
        """

        super(LinearInverseModel, self).__init__()

        self.activation = activation

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_1 = init_layer(nn.Linear(in_dim, hidden_size))
        self.inv_2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.inv_3 = init_layer(nn.Linear(hidden_size, out_dim),
            weight_std=out_init)

    def forward(self,
                enc_obs_1,
                enc_obs_2):

        obs_input = torch.cat((enc_obs_1, enc_obs_2), dim=1)

        out = self.inv_1(obs_input)
        out = self.activation(out)

        out = self.inv_2(out)
        out = self.activation(out)

        out = self.inv_3(out)
        out = F.softmax(out, dim=-1)

        return out


class LinearForwardModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 out_init,
                 act_dim,
                 hidden_size,
                 action_type,
                 activation = nn.ReLU(),
                 **kw_args):
        """
            The forward model of the ICM method. In short, this learns to
            predict the changes in state given a current state and action.
            In reality, our states can be observations, and our observations
            are typically encoded into a form that only contains information
            that an actor needs to make decisions.

            This implementation uses a simple feed-forward network.
        """

        super(LinearForwardModel, self).__init__()

        self.activation  = activation
        self.action_type = action_type
        self.act_dim     = act_dim

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.f_1 = init_layer(nn.Linear(in_dim, hidden_size))
        self.f_2 = init_layer(nn.Linear(hidden_size, hidden_size))
        self.f_3 = init_layer(nn.Linear(hidden_size, out_dim),
            weight_std=out_init)

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
        out = self.activation(out)

        out = self.f_2(out)
        out = self.activation(out)

        out = self.f_3(out)
        return out


class ICM(PPONetwork):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_type,
                 out_init        = 1.0,
                 obs_encoder     = LinearObservationEncoder,
                 reward_scale    = 0.01,
                 activation      = nn.ReLU(),
                 encoded_obs_dim = 128,
                 hidden_size     = 128,
                 **kw_args):
        """
            The Intrinsic Curiosit Model (ICM).
        """

        super(ICM, self).__init__(**kw_args)

        self.act_dim      = act_dim
        self.reward_scale = reward_scale
        self.action_type  = action_type

        self.activation   = activation
        self.ce_loss      = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss     = nn.MSELoss(reduction="none")

        #
        # Observation encoder.
        #
        self.obs_encoder = obs_encoder(
            obs_dim,
            encoded_obs_dim,
            out_init,
            hidden_size,
            **kw_args)

        #
        # Inverse model; Predict the a_1 given s_1 and s_2.
        #
        self.inv_model = LinearInverseModel(
            encoded_obs_dim * 2, 
            act_dim,
            out_init,
            hidden_size,
            **kw_args)

        #
        # Forward model; Predict s_2 given s_1 and a_1.
        #
        self.forward_model = LinearForwardModel(
            encoded_obs_dim + act_dim,
            encoded_obs_dim,
            out_init,
            act_dim,
            hidden_size,
            action_type,
            **kw_args)

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
            actions = actions.reshape(action_pred.shape)
            inv_loss = self.mse_loss(action_pred, actions).mean()

        #
        # Forward model prediction.
        #
        obs_2_pred = self.forward_model(enc_obs_1, actions)
        f_loss = self.mse_loss(obs_2_pred, enc_obs_2)

        intrinsic_reward = (self.reward_scale / 2.0) * f_loss.sum(dim=-1)
        f_loss           = 0.5 * f_loss.mean()

        return intrinsic_reward, inv_loss, f_loss

