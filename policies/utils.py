from ppo_and_friends.policies.ppo_policy import PPOPolicy
from ppo_and_friends.policies.mat_policy import MATPolicy
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
    envs_per_proc,
    **kw_args):
    """
    Generate a policy from our arguments.

    Parameters:
    -----------
    policy_name: str
        The name of the policy.
    policy_class: class
        The class to use for the policy.
    actor_observation_space: gymnasium space
        The actor observation space.
    critic_observation_space: gymnasium space
        The critic observation space.
    action_space: gymnasium space
        The action space.
    test_mode: bool
        Are we in test mode?
    kw_args: dict
        Extra keyword args for the policy.

    Returns:
    --------
    PPOPolicy or derivative
        The created policy.
    """
    supported_types = [MATPolicy, PPOPolicy, None]
    if policy_class not in supported_types:
        msg  = "ERROR: policy_class is of unsupported type, "
        msg += "{policy_class}. Supported types are "
        msg += "{supported_types}."
        rank_print(msg)
        comm.Abort() 

    if type(policy_class) == type(None):
        policy_class = PPOPolicy

    policy = policy_class(
        name                      = policy_name,
        action_space              = action_space,
        actor_observation_space   = actor_observation_space,
        critic_observation_space  = critic_observation_space,
        test_mode                 = test_mode,
        envs_per_proc             = envs_per_proc,
        **kw_args)

    return policy


def get_single_policy_defaults(
    env_generator,
    policy_args,
    policy_name = "single_agent",
    agent_name  = "agent0",
    policy_type = PPOPolicy):
    """
    A convenience function for creating a single-agent policy for
    a single agent environment. This function returns the
    settings needed to pass to the trainer.

    Parameters:
    -----------
    env_generator: function
        A function mapping to an instance of our environment.
    policy_args: dict
        keyword args for the policy class.
    policy_name: str
        The name of the policy.
    agent_name: str
        The name of one of the agents of the shared policy.
    policy_type: class
        A class of type PPOPolicy (or inheriting from).

    Returns:
    --------
    tuple
        A tuple of form (policy_settings, policy_mapping_fn).
    """

    policy_settings = { policy_name : \
        (policy_type,
         env_generator().observation_space[agent_name],
         env_generator().critic_observation_space[agent_name],
         env_generator().action_space[agent_name],
         policy_args)
    }

    policy_mapping_fn = lambda *args : policy_name

    return policy_settings, policy_mapping_fn
