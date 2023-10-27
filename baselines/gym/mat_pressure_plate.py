import gym as old_gym
from ppo_and_friends.environments.gym.wrappers import MultiAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.gym.version_wrappers import Gym21ToGymnasium
import torch.nn as nn
from ppo_and_friends.utils.mpi_utils import rank_print
import pressureplate
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from ppo_and_friends.policies.mat_policy import MATPolicy

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

        import pyglet
        try:
            if version.parse(pyglet.version) > version.parse('1.5.0'):
                msg  = "WARNING: PressurePlate requires pyget version <= 1.5.0 "
                msg += "for rendering. Rendering will fail with later versions."
                rank_print(msg)
        except:
            pass

        # NOTE: the compatibility wrapper that gym comes with wasn't
        # working here...
        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21ToGymnasium(old_gym.make('pressureplate-linear-4p-v0')),
                critic_view   = "local",
                add_agent_ids = False)

        mat_kw_args  = {}

        #
        # TODO: find good setttings for MAT pressure plate.
        #
        lr = 0.0003

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 1000000,
            max_value     = 0.02,
            min_value     = 0.01)

        policy_args = {\
            "mat_kw_args"      : mat_kw_args,
            "lr"               : lr,
            "entropy_weight"   : entropy_weight,
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_type   = MATPolicy,
            policy_name   = "pressure-plate",
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     epochs_per_iter    = 15,
                     max_ts_per_ep      = 128,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     **self.kw_run_args)
