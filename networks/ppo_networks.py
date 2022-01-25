"""
    A home for PPO specific parent networks.
"""
import torch.nn as nn
import torch
from .utils import GaussianDistribution, CategoricalDistribution
import os

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
