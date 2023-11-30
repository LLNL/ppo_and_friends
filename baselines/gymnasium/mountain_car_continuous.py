import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
import numpy as np
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class MountainCarContinuousRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('MountainCarContinuous-v0',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["activation"]  =  nn.LeakyReLU()

        actor_kw_args["hidden_size"]  = 128
        actor_kw_args["hidden_depth"] = 2

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"]  = 128
        critic_kw_args["hidden_depth"] = 2

        icm_kw_args = {}
        icm_kw_args["inverse_hidden_depth"] = 2
        icm_kw_args["inverse_hidden_size"]  = 256
        icm_kw_args["forward_hidden_depth"] = 2
        icm_kw_args["forward_hidden_size"]  = 256

        lr = 0.0003

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"   : (-0.001, np.inf),
            "enable_icm"         : True,
            "icm_kw_args"        : icm_kw_args,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "natural score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(999)

        #
        # I've noticed that normalizing rewards and observations
        # can slow down learning at times. It's not by much (maybe
        # 10-50 iterations).
        #
        self.run_ppo(env_generator      = env_generator,
                     save_when          = save_when,
                     ts_per_rollout     = ts_per_rollout,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = 999,
                     ext_reward_weight  = 1./100.,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     normalize_values   = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
