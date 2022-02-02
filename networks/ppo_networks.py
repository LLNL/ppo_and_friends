"""
    A home for PPO specific parent networks.
"""
import torch.nn as nn
import torch
import sys
from .utils import GaussianDistribution, CategoricalDistribution
import os

class PPONetwork(nn.Module):
    """
        A base class for PPO networks.
    """

    def __init__(self,
                 name,
                 **kw_args):
        """
            Arguments:
                name    The name of the network.
        """

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
        """
            NOTE: if this class is to behave as a PPO actor, it should be
            given the name "actor".

            Arguments:
                action_type      Where in the observation space the split
                out_dim          The output dimension.
        """

        super(PPOActorCriticNetwork, self).__init__(**kw_args)

        if action_type not in ["discrete", "continuous"]:
            msg = "ERROR: unknown action type {}".format(action_type)
            print(msg)
            sys.exit(1)

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

    def get_result(self, obs):
        """
            Given an observation, return the results of performing
            inference + any other alterations that should be made
            to the result before it's fed back into the world.

            Arguments:
                obs    The observation to infer from.

            Returns:
                The predicted result with any required alterations.
        """
        if self.name == "actor":
            res = self.__call__(obs)
            res = self.distribution.refine_sample(res)
            return res
        else:
            return self.__call__(obs)


class PPOConv2dNetwork(PPOActorCriticNetwork):

    def __init__(self, **kw_args):
        super(PPOConv2dNetwork, self).__init__(**kw_args)


class SingleSplitObservationNetwork(PPOActorCriticNetwork):
    """
        The idea here is to support splitting the observations into
        two sub-networks before merging them back together. This is
        usually used when wanting to split proprioceptive and
        exteroceptive information.
    """

    def __init__(self,
                 split_start,
                 **kw_args):
        """
            Arguments:
                split_start      Where in the observation space the split
                                 should start.
        """

        super(SingleSplitObservationNetwork, self).__init__(**kw_args)

        if split_start <= 0:
            msg  = "ERROR: SingleSplitObservationNetwork requires a split "
            msg += "start > 0."
            print(msg)
            sys.exit(1)

        self.split_start = split_start
