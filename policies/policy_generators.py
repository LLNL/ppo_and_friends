from ppo_and_friends.policies.agent_policy import AgentPolicy
from ppo_and_friends.utils.mpi_utils import rank_print

from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def policy_generator(
    policy_class,
    observation_space,
    action_space,
    device,
    **kw_args):
    """
    """
    if policy_class not in [None, AgentPolicy]:
        msg  = "ERROR: AgentPolicy is the only currently supported policy "
        msg += "class. {} is an invalid option.".format(policy_class)
        rank_print(msg)
        comm.Abort() 

    policy = AgentPolicy(
        action_space      = action_space,
        observation_space = observation_space,
        device            = device,
        **kw_args)
