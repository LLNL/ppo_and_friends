"""
    A home for network utilities.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import numpy as np

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


def init_layer(layer,
               weight_std = np.sqrt(2),
               bias_const = 0.0):

    nn.init.orthogonal_(layer.weight, weight_std)
    nn.init.constant_(layer.bias, bias_const)

    return layer
