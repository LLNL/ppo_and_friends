import gym as old_gym
import lbforaging
from ppo_and_friends.environments.gym.wrappers import MultiAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.gym.version_wrappers import Gym21ToGymnasium
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.policies.mat_policy import MATPolicy

from mpi4py import MPI
comm      = MPI.COMM_WORLD

@ppoaf_runner
class MATLevelBasedForagingRunner(GymRunner):

    def run(self):
        if old_gym.__version__ != '0.21.0':
            msg  = "ERROR: LevelBasedForaging requires gym version 0.21. "
            msg += "It is not currently supported in gymnasium."
            rank_print(msg)
            comm.Abort()

        env_generator = lambda : \
            MultiAgentGymWrapper(Gym21ToGymnasium(
                old_gym.make('Foraging-8x8-3p-2f-v2')),
                critic_view = "local")

        mat_kw_args  = {}

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 1500,
            max_value     = 0.0003,
            min_value     = 0.0001)

        policy_args = {\
            "mat_kw_args"    : mat_kw_args,
            "lr"             : lr,
            "bootstrap_clip" : (0, 1),
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_type   = MATPolicy,
            policy_name   = "lbf",
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 256

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 10000,
                     epochs_per_iter    = 15,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     **self.kw_run_args)
