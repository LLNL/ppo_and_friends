import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.lstm import LSTMNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class CartPoleLSTMRunner(GymRunner):
    """
    This enviornment does not need LSTM to train. This is merely an example of
    how to use LSTM for an environment.
    """

    #
    # NOTE: this is just an example of how to extend the ppoaf CLI args.
    #
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
        parser.add_argument("--learning_rate", type=float, default=0.001)
        return parser

    def run(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('CartPole-v0',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {}
        actor_kw_args["activation"] = nn.LeakyReLU()
        actor_kw_args["sequence_length"]  = 5
        actor_kw_args["lstm_hidden_size"] = 32
        actor_kw_args["ff_hidden_size"]   = 16

        critic_kw_args = actor_kw_args.copy()

        policy_args = {\
            "ac_network"       : LSTMNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : self.cli_args.learning_rate,
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
