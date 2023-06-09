import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.actor_critic_networks import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AntRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Ant-v4',
                render_mode = self.get_gym_render_mode()))

        #
        # Ant observations are organized as follows:
        #    Positions: 13
        #    Velocities: 14
        #    Contact forces: 84
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.Tanh()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 100,
            max_value     = 0.00025,
            min_value     = 0.0001)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-10., 10.),
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 64,
                     epochs_per_iter    = 32,
                     ts_per_rollout     = ts_per_rollout,
                     obs_clip           = (-30., 30.),
                     reward_clip        = (-10., 10.),
                     **self.kw_run_args)
