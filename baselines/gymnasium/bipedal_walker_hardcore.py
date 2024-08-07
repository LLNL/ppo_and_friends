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
class BipedalWalkerHardcoreRunner(GymRunner):

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
        parser.add_argument("--bs_clip_min", default=-np.inf, type=float)
        parser.add_argument("--bs_clip_max", default=np.inf, type=float)

        parser.add_argument("--learning_rate", default=0.00025, type=float)
        parser.add_argument("--use_lr_scheduler", default=0, type=int)
        parser.add_argument("--lr_scheduler_status_max", default=100_000_000, type=int)

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

        parser.add_argument("--max_ts_per_ep", type=int, default=64)
        parser.add_argument("--ts_per_rollout", type=int, default=512)

        parser.add_argument("--mini_batch_size", type=int, default=512)
        return parser

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('BipedalWalkerHardcore-v3',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["std_offset"]  = 0.1
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

        #
        # This environment is a pretty challenging one and can be
        # very finicky. Learning rate and reward clipping have a
        # pretty powerfull impact on results, and it can be tricky
        # to get these right. Here's what I've found works best:
        #
        #   1. First, run the problem with a pretty standard learning
        #      rate (0.0001 works well), and use a conservative reward
        #      clipping of (-1, 10). Clipping the reward at -1 results
        #      in the agent learning a good gait pretty early on.
        #
        #   2. After a while (roughly 7000->8000 iterations), the agent should
        #      have a pretty solid policy. Running tests will show that
        #      it can regularly reach scores over 300, but averaging
        #      over 100 runs will likely be in the 200s. My impression
        #      here is that the -1 reward clipping, while allowing the
        #      agent to learn a good gait quickly, also causes the agent
        #      be less concerned with failures. So, at this point, I find
        #      that adjusting the lower bound of the clip to the standard
        #      -10 value allows the agent to learn that falling is
        #      actually really bad. I also lower the learning rate here
        #      to help with stability in this last phase. This last bit
        #      of learning can take a while (~10,000 -> 11,000 iterations).
        #
        # The above is all automated with the settings used below. I typically
        # run with 4 processors. The resulting policy can regularly reach average
        # scores of 320+ over 100 test runs.
        #
        #lr = LinearStepScheduler(
        #    status_key      = "iteration",
        #    initial_value   = 0.0001,
        #    status_triggers = [3900,],
        #    step_values     = [0.00001,])

        #reward_clip_min = LinearStepScheduler(
        #    status_key      = "iteration",
        #    initial_value   = -1.,
        #    status_triggers = [4000,],
        #    step_values     = [-10.,])

        #bs_clip_min = LinearStepScheduler(
        #    status_key      = "iteration",
        #    initial_value   = -1.,
        #    status_triggers = [4000,],
        #    step_values     = [-10.,])

        if bool(self.cli_args.use_lr_scheduler):
            lr = LinearScheduler(
                status_key = "iteration",
                status_max = self.cli_args.lr_scheduler_status_max,
                max_value  = self.cli_args.learning_rate,
                min_value  = 1e-10)
        else:
            lr = self.cli_args.learning_rate

        bs_clip_min = self.cli_args.bs_clip_min
        bs_clip_max = self.cli_args.bs_clip_max

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
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
                     epochs_per_iter    = 32,
                     max_ts_per_ep      = self.cli_args.max_ts_per_ep,
                     ts_per_rollout     = self.cli_args.ts_per_rollout,
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10, 10.),
                     **self.kw_run_args)
