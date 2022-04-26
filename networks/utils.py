"""
    A home for network utilities.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import numpy as np

# TODO: we should probably create an abstract base class to enforce and
# explain some of these decisions.
class CategoricalDistribution(object):
    """
        A module for obtaining a categorical probability distributions.

        NOTE: this module is very simple, but it has support structures
        for more complicated cases.
    """

    def __init__(self,
                 **kw_args):
        """
            Nothing to see here...
        """
        pass

    def get_distribution(self, probs):
        """
            Given a set of probabilities, create and return a
            categorical distribution.

            Arguments:
                probs    The probabilities that require a distribution.

            Returns:
                A PyTorch Categorical distribution object.
        """
        return Categorical(probs)

    def get_log_probs(self, dist, actions):
        """
            Get the log probabilities from a distribution and
            a set of actions.

            Arguments:
                dist        The distribution.
                actions     The actions to find the log probs of.

            Returns:
                The log probabilities of the given actions from the
                given distribution.
        """
        return dist.log_prob(actions)

    def sample_distribution(self, dist):
        """
            Given a distribution, return a sample from that distribution.
            Trick business: some distributions will alter one of the
            returned samples. In that case, we still need access to the
            original sample. This distribution does not perform any
            alterations, so we just return the same sample twice.

            Arguments:
                dist    The distribution to sample from.

            Returns:
                A tuple of form (sample, sample), where each item
                is an identical sample from the distribution.
        """
        sample = dist.sample()
        return sample, sample

    def get_entropy(self, dist, _):
        """
            Get the entropy of a categorical distribution.

            Arguments:
                dist    The distribution to get the entropy of.

            Returns:
                The distributions entropy.
        """
        return dist.entropy()

    def refine_sample(self,
                      sample,
                      testing = False):
        """
            Given a sample from the distribution, refine this
            sample. In our case, we have no refinements yet.

            Arguments:
                sample    The sample to refine.

            Returns:
                The refined sample.
        """
        return sample


class GaussianDistribution(nn.Module):
    """
        A module for obtaining a Gaussian distribution.

        This distribution will learn the log_std from training.
        arXiv:2006.05990v1 suggests that learning this in the
        network or separately doesn't really make a difference.
        This is a bit simpler.
    """

    def __init__(self,
                 act_dim,
                 std_offset       = 0.5,
                 min_std          = 0.01,
                 distribution_min = -1.,
                 distribution_max = 1.,
                 **kw_args):
        """
            Arguments:
                act_dim           The action dimension.
                std_offset        An offset to use when initializing the log
                                  std. It will be negated.
                min_std           A minimum action std to enforce.
                distribution_min  A lower bound to enforce in the distribution.
                distribution_max  An upper bound to enforce in the distribution.
        """
        super(GaussianDistribution, self).__init__()

        self.min_std  =  torch.tensor([min_std]).float()
        self.dist_min = distribution_min
        self.dist_max = distribution_max

        #
        # arXiv:2006.05990v1 suggests an offset of -0.5 is best for
        # most continuous control tasks, but there are some which perform
        # better with higher values. I've found some environments perform
        # much better with lower values.
        #
        log_std = torch.as_tensor(-std_offset * np.ones(act_dim, dtype=np.float32))
        self.log_std = torch.nn.Parameter(log_std)

    def get_distribution(self, action_mean):
        """
            Given an action mean or batch of action means, return
            gaussian distributions.

            Arguments:
                action_mean    A single isntance of batch of action means.

            Returns:
                A Gaussian distribution.
        """

        #
        # arXiv:2006.05990v1 suggests that softplus can perform
        # slightly better than exponentiation.
        # TODO: add option to use softplus or exp.
        #
        std = nn.functional.softplus(self.log_std)
        std = torch.max(std.cpu(), self.min_std)
        return Normal(action_mean, std)

    def get_log_probs(self,
                      dist,
                      pre_tanh_actions,
                      epsilon = 1e-6):
        """
            Given a Gaussian distribution and "raw" (pre-tanh) actions,
            return the log probabilities of those actions.

            Arguments:
                dist                The Guassian distribution to query.
                pre_tanh_actions    A set of "raw" actions, i.e. actions
                                    that have not been squashed using Tanh
                                    (or any function).
                epsilon             A small number to help with avoiding
                                    zero-divisions.

            Returns:
                The log probabilities of the given actions.
        """
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

    def _enforce_sample_range(self, sample):
        """
            Force a given sample into the range [dist_min, dist_max].

            Arguments:
                sample    The sample to alter into the above range.

            Returns:
                The input sample after being transformed into the
                range [dist_min, dist_max].
        """
        #
        # We can use a simple interpolation:
        #     new_sample <-
        #       ( (new_max - new_min) * (sample - current_min) ) /
        #       (current_max - current_min) + new_min
        #
        sample = ((sample + 1.) / 2.) * \
            (self.dist_max - self.dist_min) + self.dist_min
        return sample

    def refine_sample(self,
                      sample,
                      testing = False):
        """
            Given a sample from the distribution, refine this
            sample. In our case, this means checking if we need
            to enforce a particular range.

            Arguments:
                sample      The sample to refine.
                testing     If we are testing, it seems that we need to send
                            our sample through tanh before enforcing a range.
                            The testing performs much better in this case.

            Returns:
                The refined sample.
        """
        if self.dist_min != -1.0 or self.dist_max != 1.0:

            #
            # NOTE:
            # The humanoid env is the only one (so far) that requires an
            # enforced range.
            # In this env, sending the sample through tanh before enforcing a
            # range results in improved performance. However, I've found that
            # there are some envs that perform much worse when the sample is
            # sent through tanh during testing, and others that seem to be
            # unaffected. Given this, I'm only using tanh if a range is
            # enforced. Let's keep an eye on this.
            #
            if testing:
                sample = torch.tanh(sample)

            sample = self._enforce_sample_range(sample)

        return sample

    def sample_distribution(self, dist):
        """
            Sample a Gaussian distribution. In this case, we
            want to return two different versions of the sample:
                1. The un-altered, raw sample.
                2. A version of the sample that has been sent though
                   a Tanh function, i.e. squashed to a [-1, 1] range,
                   and potentially altered further to fit a different
                   range.

            Arguments:
                dist    The Gaussian distribution to sample.

            Returns:
                A tuple of form (tanh_sample, raw_sample).
        """
        sample      = dist.sample()
        tanh_sample = torch.tanh(sample)
        refined     = self.refine_sample(tanh_sample)

        return refined, sample

    def get_entropy(self,
                    dist,
                    pre_tanh_actions,
                    epsilon = 1e-6):
        """
            Given a Gaussian distribution, calculate the entropy of
            a set of pre-tanh actions, i.e. raw actions that have
            not been altered by a tanh (or any other) function.

            Arguments:
                dist                The Guassian distribution to query.
                pre_tanh_actions    A set of "raw" actions, i.e. actions
                                    that have not been squashed using Tanh
                                    (or any function).
                epsilon             A small number to help with avoiding
                                    zero-divisions.

            Returns:
                The entropy of the given actions.
        """
        #
        # This is a bit odd here... arXiv:2006.05990v1 suggests using
        # tanh to move the actions into a [-1, 1] range, but this also
        # changes the probability densities. They suggest it is okay for most
        # situations because if the differntiation (see above comments),
        # but it does affect the entropy. They suggest using the equation
        # the following equation:
        #    Ex[-log(x) + log(tanh^prime (x))] s.t. x is the pre-tanh
        #    computed probability distribution.
        # Note that this is the same as our log probs when using tanh but
        # negated.
        #
        return -self.get_log_probs(dist, pre_tanh_actions, epsilon)


def get_conv2d_out_size(in_size,
                        padding,
                        kernel_size,
                        stride):
    """
        Get the output size of a 2d convolutional layer.

        Arguments:
            in_size        The input size.
            padding        The padding used.
            kernel_size    The kernel size used.
            stride         The stride used.

        Returns:
            The expected output size of the convolution.
    """
    out_size = int(((in_size + 2.0 * padding - (kernel_size - 1) - 1)\
        / stride) + 1)
    return out_size


def get_maxpool2d_out_size(in_size,
                           padding,
                           kernel_size,
                           stride):
    """
        Get the output size of a 2d max pool layer.

        Arguments:
            in_size        The input size.
            padding        The padding used.
            kernel_size    The kernel size used.
            stride         The stride used.

        Returns:
            The expected output size of the max pool.
    """
    return get_conv2d_out_size(in_size, padding, kernel_size, stride)


def init_layer(layer,
               weight_std = np.sqrt(2),
               bias_const = 0.0):
    """
        Orthogonally initialize a neural network layer using an std weight and
        a bias constant.

        Arguments:
            layer         The network layer.
            weight_std    The std weight.
            bias_const    The bias constants.

        Returns:
            The initialized layer.
    """

    nn.init.orthogonal_(layer.weight, weight_std)
    nn.init.constant_(layer.bias, bias_const)

    return layer

def init_net_parameters(net,
                        weight_std = np.sqrt(2),
                        bias_const = 0.0):
    """
        Orthogonally initialize a neural network using an std weight and
        a bias constant.

        Arguments:
            net           The network.
            weight_std    The std weight.
            bias_const    The bias constants.

        Returns:
            The initialized network.
    """

    for name, param in net.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param, weight_std)
        elif 'bias' in name:
            nn.init.constant_(param, bias_const)

    return net
