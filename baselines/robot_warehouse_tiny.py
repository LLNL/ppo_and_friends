import gym as old_gym
import rware
from ppo_and_friends.environments.gym.wrappers import MultiAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.gym.version_wrappers import Gym21ToGymnasium
import torch.nn as nn
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.runners.runner_tags import ppoaf_runner

from mpi4py import MPI
comm      = MPI.COMM_WORLD

@ppoaf_runner
class RobotWarehouseTinyRunner(GymRunner):

    def run(self):
        if old_gym.__version__ != '0.21.0':
            msg  = "ERROR: RobotWarehouse requires gym version 0.21. "
            msg += "It is not currently supported in gymnasium."
            rank_print(msg)
            comm.barrier()
            comm.Abort()

        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21ToGymnasium(old_gym.make('rware-tiny-3ag-v1')),
                critic_view = "policy",
                policy_mapping_fn = lambda *args : "rware",
                add_agent_ids = True)

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 1000000,
            max_value     = 0.0003,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 1000000,
            max_value     = 0.015,
            min_value     = 0.01)

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"     : (0., 10.),
            "entropy_weight"     : entropy_weight,
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "rware",
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(1024)

        #
        # This environment comes from arXiv:2006.07869v4.
        # This is a very sparse reward environment, and there are series of
        # complex actions that must occur in between rewards. Because of this,
        # using a large maximum timesteps per episode results in faster learning.
        # arXiv:2103.01955v2 suggests using smaller epoch counts for complex
        # environments and large batch sizes (single batches if possible).
        # Because of the sparse rewards, I've also increased the entropy
        # weight to incentivize exploration. We could also experiment with
        # using ICM here. I've disabled rewards and observation normalization
        # and clipping, mainly because they aren't mentioned in arXiv:2103.01955v2.
        # I've noticed that performance tends to go down a bit when these
        # normalizations are enabled.
        #
        self.run_ppo(
                env_generator      = env_generator,
                policy_settings    = policy_settings,
                policy_mapping_fn  = policy_mapping_fn,
                batch_size         = 10000,
                epochs_per_iter    = 5,
                max_ts_per_ep      = 32,
                ts_per_rollout     = ts_per_rollout,
                normalize_obs      = False,
                obs_clip           = None,
                normalize_rewards  = False,
                reward_clip        = None,
                **self.kw_run_args)
