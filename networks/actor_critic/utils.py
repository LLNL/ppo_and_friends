from ppo_and_friends.networks.distributions import *
import torch.nn.functional as t_functional
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype, get_space_shape
from ppo_and_friends.utils.misc import get_flattened_space_length

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def get_actor_distribution(
    action_space,
    **kw_args):
    """
    """
    action_dtype    = get_action_dtype(action_space)
    output_func = lambda x : x

    if action_dtype not in ["discrete", "continuous",
        "multi-binary", "multi-discrete"]:

        msg = "ERROR: unknown action type {}".format(action_dtype)
        rank_print(msg)
        comm.Abort()

    if action_dtype == "discrete":
        distribution = CategoricalDistribution(**kw_args)
        output_func  = lambda x : t_functional.softmax(x, dim=-1)
    
    if action_dtype == "multi-discrete":
        distribution = MultiCategoricalDistribution(
            nvec = action_space.nvec, **kw_args)
        output_func  = lambda x : t_functional.softmax(x, dim=-1)
    
    elif action_dtype == "continuous":
        out_size = get_flattened_space_length(action_space)
        distribution = GaussianDistribution(out_size, **kw_args)
    
    elif action_dtype == "multi-binary":
        distribution = BernoulliDistribution(**kw_args)
        output_func  = t_functional.sigmoid

    return distribution, output_func
