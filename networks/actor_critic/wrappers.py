from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_space_shape, get_action_prediction_shape
from ppo_and_friends.networks.distributions import get_actor_distribution

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def to_actor(ac_network):
    """
    Convert a general PPONetwork to an ActorNetwork.
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
                out_shape = get_action_prediction_shape(action_space),
                **kw_args)

            self.obs_space = obs_space
            self.distribution, self.output_func = \
                get_actor_distribution(action_space, **kw_args)
    
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
    Convert a general PPONetwork to a CriticNetwork.
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
