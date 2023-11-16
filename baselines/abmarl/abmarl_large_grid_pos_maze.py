from ppo_and_friends.environments.abmarl.wrappers import AbmarlWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import EnvironmentRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.abmarl.envs.maze_env import lg_abmarl_grid_pos_maze
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AbmarlLargeGridPosMazeRunner(EnvironmentRunner):

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

        parser.add_argument("--learning_rate", default=5e-4, type=float)

        parser.add_argument("--enable_icm", type=int, default=0)
        parser.add_argument("--icm_inverse_size", type=int, default=32)
        parser.add_argument("--icm_inverse_depth", type=int, default=2)
        parser.add_argument("--icm_forward_size", type=int, default=32)
        parser.add_argument("--icm_forward_depth", type=int, default=2)
        parser.add_argument("--icm_encoded_obs_dim", type=int, default=9)
        parser.add_argument("--icm_learning_rate", type=float, default=0.0003)
        parser.add_argument("--intr_reward_weight", type=float, default=1.0)
        return parser

    def run(self):
        #
        # See AbmarlBlindLargeMaze for details about this environment. I spent
        # more time solving the blind case, which is more difficult, but they
        # are otherwise very similar.
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"]      = self.cli_args.icm_encoded_obs_dim
        icm_kw_args["inverse_hidden_depth"] = self.cli_args.icm_inverse_depth
        icm_kw_args["inverse_hidden_size"]  = self.cli_args.icm_inverse_size
        icm_kw_args["forward_hidden_depth"] = self.cli_args.icm_forward_depth
        icm_kw_args["forward_hidden_size"]  = self.cli_args.icm_forward_size

        #
        # Decay entropy as our policy converges.
        #
        entropy_weight = LinearStepScheduler(
            status_key      = "longest run",
            initial_value   = 0.04,
            compare_fn      = np.less_equal,
            status_triggers = [4000, 3000, 2000, 500],
            step_values     = [1e-2, 1e-3, 1e-4, 0.0])

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
            "entropy_weight"     : entropy_weight,
        }

        policy_name = "abmarl-maze"
        policy_mapping_fn = lambda *args : policy_name

        env_generator = lambda : \
                AbmarlWrapper(env               = lg_abmarl_grid_pos_maze,
                              policy_mapping_fn = policy_mapping_fn)

        policy_settings = { policy_name : \
            (None,
             env_generator().observation_space["navigator"],
             env_generator().critic_observation_space["navigator"],
             env_generator().action_space["navigator"],
             policy_args)
        }

        ts_per_rollout = self.get_adjusted_ts_per_rollout(4096)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 64,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
