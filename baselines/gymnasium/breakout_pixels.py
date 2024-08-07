import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.environments.gym.atari_wrappers import BreakoutPixelsEnvWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.convolution import AtariPixelNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class BreakoutPixelsRunner(GymRunner):

    def run(self):
        if self.kw_run_args["render"]:
            #
            # NOTE: we don't want to explicitly call render for atari games.
            # They have more advanced ways of rendering.
            #
            self.kw_run_args["render"] = False

            gym_generator = lambda : gym.make(
                    'Breakout-v4',
                    repeat_action_probability = 0.0,
                    frameskip                 = 1,
                    apply_api_compatibility   = False,
                    render_mode               = 'human')
        else:
            gym_generator = lambda : gym.make(
                    'Breakout-v0',
                    repeat_action_probability = 0.0,
                    apply_api_compatibility   = False,
                    frameskip                 = 1)

        breakout_generator = lambda : BreakoutPixelsEnvWrapper(
            env              = gym_generator(),
            allow_life_loss  = self.kw_run_args["test"],
            hist_size        = 2,
            skip_k_frames    = 4)

        env_generator = lambda : \
            SingleAgentGymWrapper(breakout_generator())

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 4000,
            max_value     = 0.0003,
            min_value     = 0.0)

        policy_args = {\
            "ac_network"       : AtariPixelNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "target_kl"        : 0.2,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     ts_per_rollout     = 512,
                     max_ts_per_ep      = 64,
                     epochs_per_iter    = 30,
                     reward_clip        = (-1., 1.),
                     normalize_obs      = False,
                     **self.kw_run_args)
