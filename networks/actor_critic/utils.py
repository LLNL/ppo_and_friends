from ppo_and_friends.networks.distributions import *
import torch.nn.functional as t_functional
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.misc import get_action_dtype
from ppo_and_friends.utils.misc import get_flattened_space_length

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

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
