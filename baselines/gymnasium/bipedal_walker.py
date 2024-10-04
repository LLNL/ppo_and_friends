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
class BipedalWalkerRunner(GymRunner):

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
        parser.add_argument("--learning_rate", default=0.00025, type=float)
        return parser

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('BipedalWalker-v3',
                render_mode = self.get_gym_render_mode()))

        #
        # The lidar observations are the last 10.
        #
        actor_kw_args = {}

        #
        # I've found that a lower std offset greatly improves performance
        # stability in this environment. Also, most papers suggest that using Tanh
        # provides the best performance, but I find that LeakyReLU works better
        # here.
        #
        actor_kw_args["std_offset"] = 0.1
        actor_kw_args["activation"] = nn.LeakyReLU()

        #
        # You can also use an LSTM or Split Observation network here,
        # but I've found that a simple MLP learns faster both in terms
        # of iterations and wall-clock time. The LSTM is the slowest
        # of the three options, which I would assume is related to the
        # fact that velocity information is already contained in the
        # observations, but it's a bit surprising that we can't infer
        # extra "history" information from the lidar.
        #
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : self.cli_args.learning_rate,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        #
        # Thresholding the reward to a low of -1 doesn't drastically
        # change learning, but it does help a bit. Clipping the bootstrap
        # reward to the same range seems to help with stability.
        #
        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     batch_size          = 512,
                     max_ts_per_ep       = 32,
                     ts_per_rollout      = 512,
                     normalize_adv       = True,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_run_args)
