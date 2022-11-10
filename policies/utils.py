from ppo_and_friends.policies.agent_policy import AgentPolicy
from ppo_and_friends.utils.mpi_utils import rank_print

from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def generate_policy(
    policy_name,
    policy_class,
    actor_observation_space,
    critic_observation_space,
    action_space,
    test_mode,
    **kw_args):
    """
    """
    if policy_class not in [None, AgentPolicy]:
        msg  = "ERROR: AgentPolicy is the only currently supported policy "
        msg += "class. {} is an invalid option.".format(policy_class)
        rank_print(msg)
        comm.Abort() 

    policy = AgentPolicy(
        name                      = policy_name,
        action_space              = action_space,
        actor_observation_space   = actor_observation_space,
        critic_observation_space  = critic_observation_space,
        test_mode                 = test_mode,
        **kw_args)

    return policy


def get_single_agent_policy_defaults(
    env_generator,
    policy_args,
    policy_name = "single_agent"):
    """
    """

    policy_settings = { policy_name : \
        (None,
         env_generator().observation_space["agent0"],
         env_generator().observation_space["agent0"],
         env_generator().action_space["agent0"],
         policy_args)
    }

    policy_mapping_fn = lambda *args : policy_name

    return policy_settings, policy_mapping_fn


#FIXME: create a policy spec?
#def PolicySpec
