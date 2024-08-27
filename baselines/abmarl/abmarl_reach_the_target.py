from ppo_and_friends.environments.abmarl.wrappers import AbmarlWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import EnvironmentRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.abmarl.envs.reach_the_target import abmarl_rtt_env
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from ppo_and_friends.policies.mat_policy import MATPolicy

@ppoaf_runner
class AbmarlReachTheTargetRunner(EnvironmentRunner):

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
        parser.add_argument("--policy", default='mappo', type=str, choices=["mat", "mappo"])
        parser.add_argument("--learning_rate", default=0.0003, type=float)
        return parser

    def run(self):

        runner_actor_kw_args  = {}
        runner_critic_kw_args = {}
        runner_mat_kw_args    = {}
        runner_policy_class   = None

        if self.cli_args.policy == "mappo":
            runner_actor_kw_args = {}
            runner_actor_kw_args["activation"]  = nn.LeakyReLU()
            runner_actor_kw_args["hidden_size"] = 64

            runner_critic_kw_args = runner_actor_kw_args.copy()
            runner_critic_kw_args["hidden_size"] = 128

        elif self.cli_args.policy == "mat":
            runner_mat_kw_args  = {}
            runner_policy_class = MATPolicy

        runner_policy_args = {\
            #
            # Only used when MAPPO is enabled.
            #
            "actor_kw_args"      : runner_actor_kw_args,
            "critic_kw_args"     : runner_critic_kw_args,

            "lr"                 : self.cli_args.learning_rate,

            #
            # Only used when MAT is enabled.
            #
            "mat_kw_args"        : runner_mat_kw_args,

            "bootstrap_clip"     : (-10.0, 100),
        }

        target_actor_kw_args = {}
        target_actor_kw_args["activation"]  = nn.LeakyReLU()
        target_actor_kw_args["hidden_size"] = 64

        target_critic_kw_args = target_actor_kw_args.copy()
        target_critic_kw_args["hidden_size"] = 128

        target_policy_args = {\
            #
            # The target is a single agent, so it ony uses PPO.
            #
            "actor_kw_args"      : target_actor_kw_args,
            "critic_kw_args"     : target_critic_kw_args,

            "lr"                 : self.cli_args.learning_rate,

            "bootstrap_clip"     : (-10.0, 100),
        }

        def policy_mapping_fn(agent_id):
            return 'runner' if agent_id.startswith('runner') else 'target'

        env_generator = lambda : \
                AbmarlWrapper(env               = abmarl_rtt_env,
                              policy_mapping_fn = policy_mapping_fn,
                              critic_view       = "policy",
                              death_mask_reward = -1.)

        #
        # This environment is multi-agent and requires different policies.
        #
        policy_settings = {
            "target" :
            (None,
             env_generator().observation_space["target"],
             env_generator().critic_observation_space["target"],
             env_generator().action_space["target"],
             target_policy_args),

            "runner" :
            (runner_policy_class,
             env_generator().observation_space["runner0"],
             env_generator().critic_observation_space["runner0"],
             env_generator().action_space["runner0"],
             runner_policy_args)
        }

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = 256,
                     **self.kw_run_args)
