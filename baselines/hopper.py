import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class HopperRunner(GymRunner):

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Hopper-v3',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.Tanh()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = 0.0003,
            status_triggers = [400,],
            step_values     = [0.0001,])

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : 0.0,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(1024)

        #
        # I find that value normalization hurts the hopper environment training.
        # That may be a result of it's combination with other settings in here.
        #
        self.run_ppo(
                env_generator      = env_generator,
                policy_settings    = policy_settings,
                policy_mapping_fn  = policy_mapping_fn,
                batch_size         = 512,
                max_ts_per_ep      = 16,
                epochs_per_iter    = 8,
                ts_per_rollout     = ts_per_rollout,
                normalize_values   = False,
                obs_clip           = (-10., 10.),
                reward_clip        = (-10., 10.),
                **self.kw_run_args)
