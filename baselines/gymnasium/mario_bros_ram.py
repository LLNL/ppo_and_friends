import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.environments.gym.atari_wrappers import RAMHistEnvWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
import numpy as np
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class MarioBrosRAMRunner(GymRunner):

    def add_cli_args(self, parser):
        """
        Define extra args that will be added to the ppoaf command.

        Parameters:
        -----------
        parser: argparse.ArgumentParser
            The parser from ppoaf.

        Returns:
        --------
        argparse.ArgumentParser:
            The same parser as the input with potentially new arguments added.
        """
        parser.add_argument("--bs_clip_min", default=-10_000_000, type=float)
        parser.add_argument("--bs_clip_max", default=10_000_000, type=float)

        parser.add_argument("--reward_clip_min", default=-1, type=float)
        parser.add_argument("--reward_clip_max", default=1, type=float)

        parser.add_argument("--learning_rate", default=0.001, type=float)

        parser.add_argument("--hist_size", default=2, type=int)

        parser.add_argument("--enable_icm", type=int, default=0)
        parser.add_argument("--icm_inverse_size", type=int, default=32)
        parser.add_argument("--icm_inverse_depth", type=int, default=2)
        parser.add_argument("--icm_forward_size", type=int, default=32)
        parser.add_argument("--icm_forward_depth", type=int, default=2)
        parser.add_argument("--icm_encoded_obs_dim", type=int, default=9)
        parser.add_argument("--icm_learning_rate", type=float, default=0.0003)
        parser.add_argument("--intr_reward_weight", type=float, default=1.0)

        parser.add_argument("--actor_hidden", type=int, default=256)
        parser.add_argument("--critic_hidden_mult", type=int, default=2)

        parser.add_argument("--max_ts_per_ep", type=int, default=16)
        parser.add_argument("--ts_per_rollout", type=int, default=512)

        parser.add_argument("--mini_batch_size", type=int, default=128)
        return parser

    def run(self):
        if self.kw_run_args["render"]:
            #
            # NOTE: we don't want to explicitly call render for atari games.
            # They have more advanced ways of rendering.
            #
            self.kw_run_args["render"] = False

            gym_generator = lambda : gym.make(
                "ALE/MarioBros-ram-v5",
                frameskip                 = 1,
                repeat_action_probability = 0.0,
                render_mode = "human")
        else:
            gym_generator = lambda : gym.make(
                "ALE/MarioBros-ram-v5",
                frameskip                 = 1,
                repeat_action_probability = 0.0)

        mario_generator = lambda : RAMHistEnvWrapper(
            env              = gym_generator(),
            allow_life_loss  = self.kw_run_args["test"],
            hist_size        = self.cli_args.hist_size,
            skip_k_frames    = 4)

        env_generator = lambda : SingleAgentGymWrapper(mario_generator())

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = self.cli_args.actor_hidden

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = self.cli_args.actor_hidden * self.cli_args.critic_hidden_mult

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"]      = self.cli_args.icm_encoded_obs_dim
        icm_kw_args["inverse_hidden_depth"] = self.cli_args.icm_inverse_depth
        icm_kw_args["inverse_hidden_size"]  = self.cli_args.icm_inverse_size
        icm_kw_args["forward_hidden_depth"] = self.cli_args.icm_forward_depth
        icm_kw_args["forward_hidden_size"]  = self.cli_args.icm_forward_size

        bs_clip_min = self.cli_args.bs_clip_min
        bs_clip_max = self.cli_args.bs_clip_max

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : self.cli_args.learning_rate,
            "enable_icm"         : self.cli_args.enable_icm,
            "icm_kw_args"        : icm_kw_args,
            "icm_lr"             : self.cli_args.icm_learning_rate,
            "bootstrap_clip"     : (bs_clip_min, bs_clip_max),
            "intr_reward_weight" : self.cli_args.intr_reward_weight,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = self.cli_args.mini_batch_size,
                     ts_per_rollout     = self.cli_args.ts_per_rollout,
                     max_ts_per_ep      = self.cli_args.max_ts_per_ep,
                     epochs_per_iter    = 30,
                     reward_clip        = (self.cli_args.reward_clip_min, self.cli_args.reward_clip_max),
                     normalize_obs      = False,
                     normalize_rewards  = True,
                     **self.kw_run_args)
