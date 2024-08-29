from pettingzoo.mpe import simple_spread_v3
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.environments.petting_zoo.wrappers import ParallelZooWrapper
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
import ppo_and_friends.networks.actor_critic.multi_agent_transformer as mat
from ppo_and_friends.policies.mat_policy import MATPolicy

@ppoaf_runner
class MPESimpleSpreadRunner(GymRunner):

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
        parser.add_argument("--continuous_actions", default=0, choices=[0, 1], type=int)
        parser.add_argument("--learning_rate", default=0.0003, type=float)
        return parser

    def run(self):

        policy_map = lambda x: 'agent'

        env_generator = lambda : \
            ParallelZooWrapper(
                simple_spread_v3.parallel_env(
                    N=3,
                    local_ratio=0.5,
                    max_cycles=64,
                    continuous_actions=bool(self.cli_args.continuous_actions),
                    render_mode=self.get_gym_render_mode()),

                add_agent_ids     = False,
                critic_view       = "policy" if self.cli_args.policy == "mappo" else "local",
                policy_mapping_fn = policy_map)

        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        mat_kw_args = {}
        mat_kw_args["embedding size"] = 64

        if self.cli_args.policy == "mat":
            policy_class = MATPolicy
            ac_network   = mat.MATActorCritic
        else:
            policy_class = None
            ac_network   = FeedForwardNetwork

        policy_args = {\
            "ac_network"       : ac_network,

            #
            # Only used if MAT is disabled.
            #
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,

            #
            # Only used if MAT is enabled.
            #
            "mat_kw_args"      : mat_kw_args,

            "lr"               : self.cli_args.learning_rate,
        }

        policy_settings = { 
            "agent" : \
                (policy_class,
                 env_generator().observation_space["agent_0"],
                 env_generator().critic_observation_space["agent_0"],
                 env_generator().action_space["agent_0"],
                 policy_args),
        }

        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_map,
                     max_ts_per_ep       = 64,
                     ts_per_rollout      = 256,
                     batch_size          = 128,
                     normalize_obs       = False,
                     obs_clip            = None,
                     normalize_rewards   = False,
                     reward_clip         = None,
                     **self.kw_run_args)
