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
        Generate a policy from our arguments.

        Arguments:
            policy_name               The name of the policy.
            policy_class              The class to use for the policy.
            actor_observation_space   The actor observation space.
            critic_observation_space  The critic observation space.
            action_space              The action space.
            test_mode                 Are we in test mode (bool)?
            kw_args                   Extra keyword args for the policy.

        Returns:
            The created policy.
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


def get_single_policy_defaults(
    env_generator,
    policy_args,
    policy_name = "single_agent",
    agent_name  = "agent0"):
    """
        A convenience function for creating a single-agent policy for
        a single agent environment. This function returns the
        settings needed to pass to the trainer.

        Arguments:
            env_generator    A function mapping to an instance of our
                             environment.
            policy_args      keyword args for the policy class.
            policy_name      The name of the policy.
            agent_name       The name of one of the agents of the shared
                             policy.

        Returns:
            A tuple of form (policy_settings, policy_mapping_fn).
    """

    policy_settings = { policy_name : \
        (None,
         env_generator().observation_space[agent_name],
         env_generator().critic_observation_space[agent_name],
         env_generator().action_space[agent_name],
         policy_args)
    }

    policy_mapping_fn = lambda *args : policy_name

    return policy_settings, policy_mapping_fn
