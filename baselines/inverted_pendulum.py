import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.actor_critic_networks import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn

class InvertedPendulumRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('InvertedPendulum-v4',
                render_mode = self.get_gym_render_mode()))

        policy_args = {\
            "ac_network" : FeedForwardNetwork,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     ts_per_rollout     = ts_per_rollout,
                     **self.kw_run_args)
