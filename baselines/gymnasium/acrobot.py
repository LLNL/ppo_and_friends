import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AcrobotRunner(GymRunner):

    def run(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Acrobot-v1',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = {}
        critic_kw_args["hidden_size"] = 128

        lr = LinearScheduler(
            status_key = "iteration",
            status_max = 2000,
            max_value  = 0.0003,
            min_value  = 0.0)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = 512,
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     **self.kw_run_args)
