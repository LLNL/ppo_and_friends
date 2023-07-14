import gym as old_gym
from ppo_and_friends.environments.gym.wrappers import MultiAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.actor_critic_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.gym.version_wrappers import Gym21ToGymnasium
import torch.nn as nn
from ppo_and_friends.utils.mpi_utils import rank_print
import pressureplate
from ppo_and_friends.runners.runner_tags import ppoaf_runner

from mpi4py import MPI
comm      = MPI.COMM_WORLD

@ppoaf_runner
class PressurePlateRunner(GymRunner):

    def run(self):
        if old_gym.__version__ != '0.21.0':
            msg  = "ERROR: PressurePlate requires gym version 0.21. "
            msg += "It is not currently supported in gymnasium."
            rank_print(msg)
            comm.barrier()
            comm.Abort()

        # NOTE: the compatibility wrapper that gym comes with wasn't
        # working here...
        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21ToGymnasium(old_gym.make('pressureplate-linear-4p-v0')))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 400000,
            max_value     = 0.001,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 400000,
            max_value     = 0.02,
            min_value     = 0.0)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : entropy_weight,
            "bootstrap_clip"   : (-3, 0),
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "pressure-plate",
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key  = "longest run",
            compare_fn  = np.less_equal,
            persistent  = True)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     save_when          = save_when,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 10000,
                     epochs_per_iter    = 5,
                     max_ts_per_ep      = 128,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     soft_resets        = True,
                     **self.kw_run_args)
