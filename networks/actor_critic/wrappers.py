from ppo_and_friends.networks.distributions import *
import torch.nn.functional as t_functional
import sys
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype, get_space_shape
from ppo_and_friends.utils.misc import get_flattened_space_length

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


def to_actor(ac_network):
    """
        Convert a PPONetwork to an ActorNetwork.
    """

    class ActorNetwork(ac_network):
    
        def __init__(self, obs_space, action_space, **kw_args):
            """
                Initialize the actor network.

                Parameters
                ----------
                obs_space: gymnasium space
                    The observation space of the actor.
                action_space: gymnasium space
                    The action space of the actor.
            """
            super(ActorNetwork, self).__init__(
                in_shape  = get_space_shape(obs_space),
                out_shape = get_space_shape(action_space),
                **kw_args)

            self.obs_space = obs_space
            action_dtype   = get_action_dtype(action_space)

            if action_dtype not in ["discrete", "continuous",
                "multi-binary", "multi-discrete"]:

                msg = "ERROR: unknown action type {}".format(action_dtype)
                rank_print(msg)
                comm.Abort()

            if action_dtype == "discrete":
                self.distribution = CategoricalDistribution(**kw_args)
                self.output_func  = lambda x : t_functional.softmax(x, dim=-1)
        
            if action_dtype == "multi-discrete":
                self.distribution = MultiCategoricalDistribution(
                    nvec = action_space.nvec, **kw_args)
                self.output_func  = lambda x : t_functional.softmax(x, dim=-1)
        
            elif action_dtype == "continuous":
                out_size = get_flattened_space_length(action_space)
                self.distribution = GaussianDistribution(out_size, **kw_args)
        
            elif action_dtype == "multi-binary":
                self.distribution = BernoulliDistribution(**kw_args)
                self.output_func  = t_functional.sigmoid
    
        def get_refined_prediction(self, obs):
            """
                Send an actor's predicted probabilities through its
                distribution's refinement method.
        
                Parameters
                ----------
                obs: gymnasium space
                    The observation to infer from.
        
                Returns
                -------
                float 
                    The predicted result sent through the distribution's
                    refinement method.
            """
            res = self.__call__(obs)
            res = self.distribution.refine_prediction(res)
            return res

    return ActorNetwork


def to_critic(ac_network):
    """
        Convert a PPONetwork to a CriticNetwork.
    """
    
    class CriticNetwork(ac_network):
        def __init__(self, obs_space, **kw_args):
            """
                Inititalize the critic network.

                Parameters
                ----------
                obs_space: gymnasium space
                    The observation space of the critic.
            """
            super(CriticNetwork, self).__init__(
                in_shape  = get_space_shape(obs_space),
                out_shape = (1,),
                **kw_args)

            self.obs_space = obs_space

    return CriticNetwork
