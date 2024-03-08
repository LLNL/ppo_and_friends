from torch.distributions import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from ppo_and_friends.utils.misc import get_flattened_space_length, get_action_prediction_shape
import torch.nn.functional as t_functional
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from ppo_and_friends.utils.misc import get_flattened_space_length, get_action_prediction_shape
import functools
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class PPODistribution(object):

    def __init__(self,
                 **kw_args):
        """
        Nothing to see here...
        """
        pass

    @abstractmethod
    def get_distribution(self, probs):
        """
        Given a set of probabilities, create and return a
        distribution.

        Parameters:
        -----------
        probs: torch tensor
            The probabilities that require a distribution.

        Returns:
        --------
        A distribution object.
        """
        return

    @abstractmethod
    def get_log_probs(self, dist, actions):
        """
        Get the log probabilities from a distribution and
        a set of actions.

        Parameters:
        -----------
        dist: torch distribution
            The distribution.
        actions: torch tensor
            The actions to find the log probs of.

        Returns:
        --------
        The log probabilities of the given actions from the
        given distribution.
        """
        return

    def sample_distribution(self, dist):
        """
        Given a distribution, return a sample from that distribution.
        Tricky business: some distributions will alter one of the
        returned samples. In that case, we still need access to the
        original sample. This distribution does not perform any
        alterations, so we just return the same sample twice.

        Paraemters:
        -----------
        dist: torch distribution
            The distribution to sample from.

        Returns:
        --------
        A tuple of form (sample, sample), where each item
        is an identical sample from the distribution.
        """
        sample = dist.sample()
        return sample, sample

    def get_entropy(self, dist, *args, **kw_args):
        """
        Get the entropy of a bernoulli distribution.

        Parameters:
        -----------
        dist: torch distribution
            The distribution to get the entropy of.

        Returns:
        --------
        The distributions entropy.
        """
        return dist.entropy()

    def refine_prediction(self,
                          prediction):
        """
        Given a prediction from our network, refine it for use in
        the environment as an action.

        NOTE: this method inhibits exploration. To allow exploration,
        the distribution must be sampled.

        Parameters:
        -----------
        prediction: float or torch tensor
            The prediction to refine.

        Returns:
        --------
        The refined prediction.
        """
        return prediction


class BernoulliDistribution(PPODistribution):
    """
    A module for obtaining a bernoulli probability distribution.

    NOTE: this module is very simple, but it has support structures
    for more complicated cases.
    """

    def get_distribution(self, probs):
        """
        Given a set of probabilities, create and return a
        bernoulli distribution.

        Parameters:
        -----------
        probs: torch tensor
            The probabilities that require a distribution.

        Returns:
        --------
        A PyTorch Bernoulli distribution object.
        """
        return Bernoulli(probs)

    def get_log_probs(self, dist, actions):
        """
        Get the log probabilities from a distribution and
        a set of actions.

        Parameters:
        -----------
        dist: torch distribution
            The distribution.
        actions: torch tensor
            The actions to find the log probs of.

        Returns:
        --------
        The log probabilities of the given actions from the
        given distribution.
        """
        return dist.log_prob(actions).sum(dim=-1)

    def refine_prediction(self, prediction):
        """
        Given a prediction from our network, refine it for use in
        the environment as an action.

        NOTE: this method inhibits exploration. To allow exploration,
        the distribution must be sampled.

        Parameters:
        -----------
        prediction: torch tensor
            The prediction to refine.

        Returns:
        --------
        The refined prediction.
        """
        prediction[prediction >= 0.5] = 1.0
        prediction[prediction < 0.5]  = 0.0
        return prediction


class CategoricalDistribution(PPODistribution):
    """
    A module for obtaining a categorical probability distribution.

    NOTE: this module is very simple, but it has support structures
    for more complicated cases.
    """

    def get_distribution(self, probs):
        """
        Given a set of probabilities, create and return a
        categorical distribution.

        Parameters:
        -----------
        probs: torch tensor
            The probabilities that require a distribution.

        Returns:
        --------
        A PyTorch Categorical distribution object.
        """
        return Categorical(probs)

    def get_log_probs(self, dist, actions):
        """
         Get the log probabilities from a distribution and
         a set of actions.

         Parameters:
         -----------
         dist: torch distribution
             The distribution.
         actions: torch tensor
             The actions to find the log probs of.

         Returns:
         --------
         The log probabilities of the given actions from the
         given distribution.
        """
        return dist.log_prob(actions)

    def refine_prediction(self, prediction):
        """
        Given a prediction from our network, refine it for use in
        the environment as an action.

        NOTE: this method inhibits exploration. To allow exploration,
        the distribution must be sampled.

        Parameters:
        -----------
        prediction: torch tensor
            The prediction to refine.

        Returns:
        --------
        The refined prediction.
        """
        prediction = torch.argmax(prediction, axis=-1)
        return prediction


class MultiCategoricalDistribution(PPODistribution):
    """
    A multi-categorical distribution for supporting MultiDiscrete
    action spaces. This is basically the same as the Categorical
    distribution, except that we create lists of distributions.

    This implementation is largely inspired by
    https://github.com/pytorch/pytorch/issues/43250
    """

    def __init__(self, nvec, **kw_args):
        """
        Parameters:
        -----------
        nvec: array-like
            The result of calling <action_space>.nvec on a
            MultiDiscrete action space. This is a numpy array
            containing the number of choices for each action.
        """
        super(MultiCategoricalDistribution, self).__init__(**kw_args)
        self.nvec = nvec

    def get_distribution(self, probs):
        """
        Given a set of probabilities, create and return a
        multi-categorical distribution, which is an array
        of categorical distributions.

        Parameters:
        -----------
        probs: torch tensor
            The probabilities that require a distribution.

        Returns:
        --------
        A numpy array of PyTorch Categorical distribution objects.
        """
        dists = []
        start = 0
        for dim in self.nvec:
            stop = start + dim

            sub_probs = probs[:, start : stop]
            dists.append(Categorical(sub_probs))

            start = stop

        return np.array(dists)

    def get_log_probs(self, dists, actions):
        """
        Get the log probabilities from an array of distributions and
        a set of actions.

        Parameters:
        -----------
        dists: torch distribution
            The distributions.
        actions: torch tensor
            The actions to find the log probs of.

        Returns:
        --------
        The log probabilities of the given actions from the
        given distributions.
        """

        #
        # The actions have shape (batch_size, num_distributions). We need to
        # grab each action and send it through its associated distribution to
        # calcualte the log probs for that action.
        #
        log_probs = []
        for idx, dist in enumerate(dists):

            dist_actions = actions[:, idx]
            log_probs.append(dist.log_prob(dist_actions))

        #
        # I believe we generally sum the log probs of each distribution
        # because we consider these to be independent actions =>
        # P(A_0 and A_1) == P(A_0) * P(A_1). arXiv:1912.11077v1 suggests
        # that there are times when we may need more sophisticated approaches
        # to handle dependencies between actions. One appraoch is to use
        # a multi-head actor network that captures dependencies between
        # the different actions. This seems like a good approach when
        # P(A_0 and A_1) == P(A_0) * P(A_1 | A_0). In either case,
        # summing the log probabilities here should hold.
        #
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def sample_distribution(self, dists):
        """
        Given a distribution, return a sample from that distribution.
        Tricky business: some distributions will alter one of the
        returned samples. In that case, we still need access to the
        original sample. This distribution does not perform any
        alterations, so we just return the same sample twice.

        Parameters:
        -----------
        dists: torch distribution
            The distributions to sample from.

        Returns:
        --------
        A tuple of form (sample, sample), where each item
        is an identical sample from the distributions.
        """
        sample = []
        for idx in range(len(dists)):
            sample.append(dists[idx].sample())

        sample = torch.stack(sample, dim=1)

        return sample, sample

    def get_entropy(self, dists, *args, **kw_args):
        """
        Get the entropy of the categorical distributions.

        Parameters:
        -----------
        dists: array-like
            The distributions to get the entropy of.

        Returns:
        --------
        The distributions entropy.
        """
        return torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)

    def refine_prediction(self, prediction):
        """
        Given a prediction from our network, refine it for use in
        the environment as an action.

        NOTE: this method inhibits exploration. To allow exploration,
        the distribution must be sampled.

        Parameters:
        -----------
        prediction: torch tensor
            The prediction to refine.

        Returns:
        --------
        The refined prediction.
        """
        #
        # Our network predicts the actions as a contiguous
        # array, and we need to break it up into individual
        # arrays associated with the discrete actions.
        #
        refined_prediction = torch.zeros(self.nvec.size).long()

        start = 0
        for idx, a_dim in enumerate(self.nvec):
            stop = start + a_dim

            refined_prediction[idx] = torch.argmax(
                prediction[:, start : stop], dim=-1)

            start = stop

        prediction = refined_prediction
        return prediction


class GaussianDistribution(nn.Module, PPODistribution):
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
        Parameters:
        -----------
        act_dim: int
            The action dimension.
        std_offset: float
            An offset to use when initializing the log
            std. It will be negated.
        min_std: float
            A minimum action std to enforce.
        distribution_min: float
            A lower bound to enforce in the distribution.
        distribution_max: float
            An upper bound to enforce in the distribution.
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

        Parameters:
        -----------
        action_mean: torch tensor
            A single instance of batch of action means.

        Returns:
        --------
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
                      pre_tanh_pred,
                      epsilon = 1e-6):
        """
        Given a Gaussian distribution and "raw" (pre-tanh) actions,
        return the log probabilities of those actions.

        Parameters:
        ----------
        dist: torch distribution
            The Guassian distribution to query.
        pre_tanh_pred: torch tensor
            A set of "raw" action predictions, i.e. actions
            that have not been squashed using Tanh
            (or any function).
        epsilon: float
            A small number to help with avoiding
            zero-divisions.

        Returns:
        --------
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
        normal_log_probs = dist.log_prob(pre_tanh_pred)
        normal_log_probs = torch.clamp(normal_log_probs, -100, 100)
        normal_log_probs = normal_log_probs.sum(dim=-1)

        tanh_prime = 1.0 - torch.pow(torch.tanh(pre_tanh_pred), 2)
        tanh_prime = torch.clamp(tanh_prime, epsilon, None)
        s_log      = torch.log(tanh_prime).sum(dim=-1)
        return normal_log_probs - s_log

    def _enforce_sample_range(self, sample):
        """
        Force a given sample into the range [dist_min, dist_max].

        Parameters:
        -----------
        sample: torch tensor or float
            The sample to alter into the above range.

        Returns:
        --------
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
                      sample):
        """
        Given a sample from the distribution, refine this
        sample. In our case, this means checking if we need
        to enforce a particular range.

        Parameters:
        ----------
        sample: torch tensor or float
            The sample to refine.

        Returns:
        --------
        The refined sample.
        """
        #
        # NOTE: I've seen peculiar behavior with adding/omitting
        # tanh. Let's keep an eye on this.
        #
        sample = torch.tanh(sample)

        if self.dist_min != -1.0 or self.dist_max != 1.0:
            sample = self._enforce_sample_range(sample)

        return sample

    def refine_prediction(self, prediction):
        """
        Given a prediction from our network, refine it for use in
        the environment as an action.

        NOTE: this method inhibits exploration. To allow exploration,
        the distribution must be sampled.

        Parameters:
        -----------
        prediction: torch tensor
            The prediction to refine.

        Returns:
        --------
        The refined prediction.
        """
        #
        # In this case, we can just rely on the refine_sample method.
        #
        return self.refine_sample(prediction)

    def sample_distribution(self, dist):
        """
        Sample a Gaussian distribution. In this case, we
        want to return two different versions of the sample:
            1. The un-altered, raw sample.
            2. A version of the sample that has been sent though
               a Tanh function, i.e. squashed to a [-1, 1] range,
               and potentially altered further to fit a different
               range.

        Parameters:
        -----------
        dist: torch distribution
            The Gaussian distribution to sample.

        Returns:
        --------
        A tuple of form (tanh_sample, raw_sample).
        """
        sample  = dist.sample()
        refined = self.refine_sample(sample)

        return refined, sample

    def get_entropy(self,
                    dist,
                    pre_tanh_pred,
                    epsilon = 1e-6):
        """
        Given a Gaussian distribution, calculate the entropy of
        a set of pre-tanh actions, i.e. raw actions that have
        not been altered by a tanh (or any other) function.

        Parameters:
        ----------
        dist: torch distribution
            The Guassian distribution to query.
        pre_tanh_pred: torch tensor
            A set of "raw" action predictions, i.e. actions
            that have not been squashed using Tanh
            (or any function).
        epsilon: float
            A small number to help with avoiding
            zero-divisions.

        Returns:
        --------
        The entropy of the given actions.
        """
        #
        # This is a bit odd here... arXiv:2006.05990v1 suggests using
        # tanh to move the actions into a [-1, 1] range, but this also
        # changes the probability densities. They suggest it is okay for most
        # situations because of the differntiation (see above comments),
        # but it does affect the entropy. They suggest using
        # the following equation:
        #    Ex[-log(x) + log(tanh^prime (x))] s.t. x is the pre-tanh
        #    computed probability distribution.
        # Note that this is the same as our log probs when using tanh but
        # negated.
        #
        return -self.get_log_probs(dist, pre_tanh_pred, epsilon)


class MixedDistribution(PPODistribution):
    """
    """

    def __init__(self, tuple_space, **kw_args):
        """
        Parameters:
        -----------
        """
        super(MixedDistribution, self).__init__(**kw_args)
        assert issubclass(type(tuple_space), spaces.Tuple)

        self.nvec         = []
        self.dist_classes = []
        self.pred_sizes = []

        for space in tuple_space:

            if issubclass(type(space), spaces.Box):

                if len(space.shape) > 1:
                    msg  = f"ERROR: MixedDistribution can only accept spaces "
                    msg += f"with flattened shapes, but it's recevied the "
                    msg += f"space: {space}"
                    rank_print(msg)
                    comm.Abort()

                self.nvec.append(get_flattened_space_length(space))
                self.pred_sizes.append(get_action_prediction_shape(space)[0])

            elif issubclass(type(space), spaces.MultiBinary):

                if len(space.shape) > 1:
                    msg  = f"ERROR: MixedDistribution can only accept spaces "
                    msg += f"with flattened shapes, but it's recevied the "
                    msg += f"space: {space}"
                    rank_print(msg)
                    comm.Abort()

                self.nvec.append(space.n)
                self.pred_sizes.append(get_action_prediction_shape(space)[0])

            elif issubclass(type(space), spaces.Discrete):
                self.nvec.append(space.n)
                self.pred_sizes.append(get_action_prediction_shape(space)[0])

            elif issubclass(type(space), spaces.MultiDiscrete):
                self.nvec.append(space.nvec.sum())
                self.pred_sizes.append(get_action_prediction_shape(space)[0])

            else:
                msg  = f"ERROR: encountered unsupported space of type "
                msg += f"{type(space)} within the MixedDistribution."
                rank_print(msg)
                comm.Abort()

            dist_class, _ = get_actor_distribution(space, **kw_args)
            self.dist_classes.append(dist_class)

        self.action_sizes = []
        for space in tuple_space:
            sample = space.sample()
            if type(sample) == np.ndarray:
                self.action_sizes.append(sample.size)
            else:
                self.action_sizes.append(1)

        assert len(self.pred_sizes) == len(self.nvec)
        assert len(self.pred_sizes) == len(self.dist_classes)

        for size in self.pred_sizes:
            assert np.issubdtype(type(size), np.integer), f"{type(size)}"

        self.nvec         = np.array(self.nvec)
        self.dist_classes = tuple(self.dist_classes)
        self.pred_size    = functools.reduce(lambda a, b : a + b, self.pred_sizes)
        self.pred_sizes   = np.array(self.pred_sizes)
        self.action_size  = functools.reduce(lambda a, b : a + b, self.action_sizes)
        self.action_sizes = np.array(self.action_sizes)

    def get_distribution(self, probs):
        """
        Given a set of probabilities, create and return a
        multi-categorical distribution, which is an array
        of categorical distributions.

        Parameters:
        -----------
        probs: torch tensor
            The probabilities that require a distribution.

        Returns:
        --------
        A numpy array of PyTorch Categorical distribution objects.
        """
        dists = []
        start = 0
        for idx, dim in enumerate(self.nvec):
            stop = start + dim

            sub_probs = probs[:, start : stop]

            next_dist = self.dist_classes[idx].get_distribution(sub_probs)
            dists.append(next_dist)

            start = stop

        return np.array(dists)

    def get_log_probs(self, dists, actions):
        """
        Get the log probabilities from an array of distributions and
        a set of actions.

        Parameters:
        -----------
        dists: torch distribution
            The distributions.
        actions: torch tensor
            The actions to find the log probs of.

        Returns:
        --------
        The log probabilities of the given actions from the
        given distributions.
        """
        assert len(dists) == len(self.nvec)

        #
        # The actions have shape (batch_size, num_distributions). We need to
        # grab each action and send it through its associated distribution to
        # calcualte the log probs for that action.
        #
        log_probs = []
        start = 0
        for idx, a_size in enumerate(self.action_sizes):
            dist = dists[idx]
            stop = start + a_size

            dist_actions = actions[:, start : stop]

            dist_lp = self.dist_classes[idx].get_log_probs(dist, dist_actions)

            if len(dist_lp.shape) == 1:
                dist_lp = dist_lp.unsqueeze(-1)

            log_probs.append(dist_lp)
            start = stop

        #
        # I believe we generally sum the log probs of each distribution
        # because we consider these to be independent actions =>
        # P(A_0 and A_1) == P(A_0) * P(A_1). arXiv:1912.11077v1 suggests
        # that there are times when we may need more sophisticated approaches
        # to handle dependencies between actions. One appraoch is to use
        # a multi-head actor network that captures dependencies between
        # the different actions. This seems like a good approach when
        # P(A_0 and A_1) == P(A_0) * P(A_1 | A_0). In either case,
        # summing the log probabilities here should hold.
        #
        return torch.cat(log_probs, dim=-1).sum(dim=-1)

    def sample_distribution(self, dists):
        """
        Given a distribution, return a sample from that distribution.
        Tricky business: some distributions will alter one of the
        returned samples. In that case, we still need access to the
        original sample. This distribution does not perform any
        alterations, so we just return the same sample twice.

        Parameters:
        -----------
        dists: torch distribution
            The distributions to sample from.

        Returns:
        --------
        A tuple of form (refined_sample, sample).
        """
        sample  = []
        refined = []

        assert len(dists) == len(self.dist_classes)

        for idx, dist in enumerate(dists):
            refined_s, unrefined_s = self.dist_classes[idx].sample_distribution(dist)

            if len(unrefined_s.shape) == 1:
                refined_s   = refined_s.unsqueeze(-1)
                unrefined_s = unrefined_s.unsqueeze(-1)

            refined.append(refined_s)
            sample.append(unrefined_s)

        sample  = torch.cat(sample, dim=-1)
        refined = torch.cat(refined, dim=-1)

        return refined, sample

    def get_entropy(self, dists, unrefined_probs, **kw_args):
        """
        Get the entropy of the categorical distributions.

        Parameters:
        -----------
        dists: array-like
            The distributions to get the entropy of.
        unrefined_probs: torch tensor
            Unrefined action probabilities.

        Returns:
        --------
        The distributions entropy.
        """
        entropy  = []

        assert len(dists) == len(self.dist_classes)

        start = 0
        for idx, dim in enumerate(self.nvec):
            dist = dists[idx]
            stop = start + dim

            sub_probs = unrefined_probs[:, start : stop]

            dist_entropy = self.dist_classes[idx].get_entropy(dist, sub_probs, **kw_args)

            if len(dist_entropy.shape) == 1:
                dist_entropy = dist_entropy.unsqueeze(-1)

            entropy.append(dist_entropy)

            start = stop

        return torch.cat(entropy, dim=-1).sum(dim=-1)

    def refine_prediction(self, prediction):
        """
        Given a prediction from our network, refine it for use in
        the environment as an action.

        NOTE: this method inhibits exploration. To allow exploration,
        the distribution must be sampled.

        Parameters:
        -----------
        prediction: torch tensor
            The prediction to refine.

        Returns:
        --------
        The refined prediction.
        """
        #
        # Our network predicts the actions as a contiguous
        # array, and we need to break it up into individual
        # arrays associated with the actions of each distribution.
        #
        refined_prediction = torch.zeros(self.pred_size)

        unref_start = 0
        ref_start   = 0

        for idx, a_dim in enumerate(self.nvec):
            unref_stop = unref_start + a_dim
            ref_stop   = ref_start + self.pred_sizes[idx]

            refined_prediction[ref_start : ref_stop] = \
                self.dist_classes.refine_prediction(
                    prediction[:, unref_start : unref_stop])

            unref_start = unref_stop
            ref_start   = ref_stop

        return refined_prediction


def get_actor_distribution(
    action_space,
    **kw_args):
    """
    Get the action distribution for an actor network.

    Parameters:
    -----------
    action_space: gymnasium space
        The action space to create a distribution for.
    kw_args: dict
        Keyword args to pass to the distribution class.

    Returns:
    --------
    tuple:
        (distribtion, output_func). output_func is the function to
        apply to the output of the actor network.
    """
    action_dtype = get_action_dtype(action_space)
    output_func  = lambda x : x

    if action_dtype not in ["discrete", "continuous",
        "multi-binary", "multi-discrete", "mixed"]:

        msg = "ERROR: unknown action type {}".format(action_dtype)
        rank_print(msg)
        comm.Abort()

    if action_dtype == "mixed":
        #FIXME:
        assert 0

    if action_dtype == "discrete":
        distribution = CategoricalDistribution(**kw_args)
        output_func  = lambda x : t_functional.softmax(x, dim=-1)
    
    if action_dtype == "multi-discrete":
        distribution = MultiCategoricalDistribution(
            nvec = action_space.nvec, **kw_args)
        output_func  = lambda x : t_functional.softmax(x, dim=-1)
    
    elif action_dtype == "continuous":
        out_size = get_flattened_space_length(action_space)
        distribution_min = kw_args.get("distribution_min")
        distribution_max = kw_args.get("distribution_max")

        if distribution_min is None:
            act_min = action_space.low.min()

            if np.isinf(act_min):
                msg  = f"ERROR: attempted to use the action min as the "
                msg += f"guassian distribution min, but the distribution min "
                msg += f"must not be inf and action min is {act_min}. "
                msg += f"Set the distribution min through the actor or MAT "
                msg += f"kw_args like so: actor_kw_args['distribution_min'] = k."
                rank_print(msg)
                comm.Abort()
            else:
                msg  = f"Setting distribution min to the action space "
                msg += f"min of {act_min}."
                rank_print(msg)
                kw_args["distribution_min"] = act_min

        if distribution_max is None:
            act_max = action_space.high.max()

            if np.isinf(act_max):
                msg  = f"ERROR: attempted to use the action max as the "
                msg += f"guassian distribution max, but the distribution max "
                msg += f"must not be inf and action max is {act_max}. "
                msg += f"Set the distribution max through the actor or MAT "
                msg += f"kw_args like so: actor_kw_args['distribution_max'] = k."
                rank_print(msg)
                comm.Abort()
            else:
                msg  = f"Setting distribution max to the action space "
                msg += f"max of {act_max}."
                rank_print(msg)
                kw_args["distribution_max"] = act_max

        distribution = GaussianDistribution(out_size, **kw_args)
    
    elif action_dtype == "multi-binary":
        distribution = BernoulliDistribution(**kw_args)
        output_func  = t_functional.sigmoid

    return distribution, output_func
