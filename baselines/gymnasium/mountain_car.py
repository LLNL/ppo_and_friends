import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class MountainCarRunner(GymRunner):

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
        parser.add_argument("--cut_off_bs_clip", default=0, type=int)

        parser.add_argument("--learning_rate", default=0.0003, type=float)

        parser.add_argument("--enable_icm", type=int, default=0)
        parser.add_argument("--icm_inverse_size", type=int, default=32)
        parser.add_argument("--icm_inverse_depth", type=int, default=2)
        parser.add_argument("--icm_forward_size", type=int, default=32)
        parser.add_argument("--icm_forward_depth", type=int, default=2)
        parser.add_argument("--icm_encoded_obs_dim", type=int, default=0)
        parser.add_argument("--icm_learning_rate", type=float, default=0.0003)
        parser.add_argument("--intr_reward_weight", type=float, default=1.0)
        return parser

    def run(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('MountainCar-v0',
                render_mode = self.get_gym_render_mode()))

        actor_kw_args = {"activation" : nn.LeakyReLU()}
        actor_kw_args["hidden_size"]  = 32
        actor_kw_args["hidden_depth"] = 2

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"]  = 64
        critic_kw_args["hidden_depth"] = 2

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"]      = self.cli_args.icm_encoded_obs_dim
        icm_kw_args["inverse_hidden_depth"] = self.cli_args.icm_inverse_depth
        icm_kw_args["inverse_hidden_size"]  = self.cli_args.icm_inverse_size
        icm_kw_args["forward_hidden_depth"] = self.cli_args.icm_forward_depth
        icm_kw_args["forward_hidden_size"]  = self.cli_args.icm_forward_size

        if self.cli_args.cut_off_bs_clip > 0:
            bs_clip_min  = LinearStepScheduler(
                status_key      = "extrinsic reward avg",
                status_preface  = "single_agent",
                initial_value   = self.cli_args.bs_clip_min,
                compare_fn      = np.greater_equal,
                status_triggers = [-140,],
                step_values     = [-np.inf,])

            bs_clip_max  = LinearStepScheduler(
                status_key      = "extrinsic reward avg",
                status_preface  = "single_agent",
                initial_value   = self.cli_args.bs_clip_max,
                compare_fn      = np.greater_equal,
                status_triggers = [-140,],
                step_values     = [np.inf,])
        else:
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

        save_when = ChangeInStateScheduler(
            status_key     = "natural score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(200)

        #
        # NOTE: This environment performs dramatically  better when
        # max_ts_per_ep is set to the total timesteps allowed by the
        # environment. It's not 100% clear to me why this is the case.
        # We should probably explore this a bit more. MountainCarContinuous
        # doesn't seem to exhibit this behavior, so it's unlikely an issue
        # with ICM.
        # Also, the extrinsic reward weight fraction is very important
        # for good performance.
        #
        self.run_ppo(env_generator      = env_generator,
                     save_when          = save_when,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     ts_per_rollout     = ts_per_rollout,
                     epochs_per_iter    = 32,
                     max_ts_per_ep      = 200,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     normalize_values   = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
