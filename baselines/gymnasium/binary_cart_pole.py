import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.environments.action_wrappers import AlternateActionEnvWrapper
from gymnasium.spaces import MultiBinary
from ppo_and_friends.runners.runner_tags import ppoaf_runner

class MultiBinaryCartPoleWrapper(AlternateActionEnvWrapper):
    """
        A simple multi-binary action version of CartPole. This is for
        testing purposes only.
    """

    def _get_action_space(self):
        return MultiBinary(1)

    def step(self, action):
        return self.env.step(int(action.item()))

@ppoaf_runner
class BinaryCartPoleRunner(GymRunner):
    """
        This is merely the original CartPole environment wrapped in a
        MultiBinary action space. It's used to test our multi-binary
        action space support.
    """

    def run(self):
        env_generator = lambda : SingleAgentGymWrapper(
            MultiBinaryCartPoleWrapper(gym.make('CartPole-v0',
                render_mode = self.get_gym_render_mode())))

        actor_kw_args = {}
        actor_kw_args["activation"] = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = 0.0002

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(**self.kw_run_args,
                     env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     ts_per_rollout     = 256,
                     max_ts_per_ep      = 32,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     normalize_adv      = True)
