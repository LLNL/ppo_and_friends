from pettingzoo.mpe import simple_adversary_v3
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
class MPESimpleAdversaryRunner(GymRunner):

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
        parser.add_argument("--icm", default=0, choices=[0, 1], type=int)
        parser.add_argument("--freeze_cycling", action="store_true",
            help="Use 'freeze cycling'.")
        return parser

    def run(self):

        policy_map = lambda name : 'adversary' if 'adversary' in name \
            else 'agent'

        env_generator = lambda : \
            ParallelZooWrapper(
                simple_adversary_v3.parallel_env(
                    N=2,
                    max_cycles=32,
                    continuous_actions=bool(self.cli_args.continuous_actions),
                    render_mode = self.get_gym_render_mode()),

                critic_view       = "policy" if self.cli_args.policy == "mappo" else "local",
                policy_mapping_fn = policy_map)

        adversary_actor_kw_args = {}

        adversary_actor_kw_args["activation"]  = nn.LeakyReLU()
        adversary_actor_kw_args["hidden_size"] = 256

        adversary_critic_kw_args = adversary_actor_kw_args.copy()
        adversary_critic_kw_args["hidden_size"] = 512

        adversary_policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : adversary_actor_kw_args,
            "critic_kw_args"   : adversary_critic_kw_args,
            "lr"               : self.cli_args.learning_rate,
            "enable_icm"       : bool(self.cli_args.icm),
        }

        agent_actor_kw_args  = {}
        agent_critic_kw_args = {}
        agent_mat_kw_args    = {}

        if self.cli_args.policy == "mat":
            agent_mat_kw_args  = {}
            agent_policy_class = MATPolicy
        else:
            agent_actor_kw_args = {}

            agent_actor_kw_args["activation"]  = nn.LeakyReLU()
            agent_actor_kw_args["hidden_size"] = 256

            agent_critic_kw_args = agent_actor_kw_args.copy()
            agent_critic_kw_args["hidden_size"] = 512

            agent_policy_class = None

        agent_policy_args =\
        {
            "lr"               : self.cli_args.learning_rate,
            "enable_icm"       : bool(self.cli_args.icm),
            "agent_shared_icm" : bool(self.cli_args.icm) and self.cli_args.policy == "mat",

            #
            # Only used if MAT is enabled.
            #
            "mat_kw_args"      : agent_mat_kw_args,

            #
            # Only used if MAPPO is enabled.
            #
            "actor_kw_args"    : agent_actor_kw_args,
            "critic_kw_args"   : agent_critic_kw_args,
        }

        policy_settings = { 
            "agent" : \
                (agent_policy_class,
                 env_generator().observation_space["agent_0"],
                 env_generator().critic_observation_space["agent_0"],
                 env_generator().action_space["agent_0"],
                 agent_policy_args),
            "adversary" : \
                (None,
                 env_generator().observation_space["adversary_0"],
                 env_generator().critic_observation_space["adversary_0"],
                 env_generator().action_space["adversary_0"],
                 adversary_policy_args),
        }

        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_map,
                     max_ts_per_ep       = 16,
                     ts_per_rollout      = 128,
                     batch_size          = 128,
                     normalize_obs       = False,
                     normalize_rewards   = False,
                     obs_clip            = (-10, 10),
                     reward_clip         = (-10, 10),
                     **self.kw_run_args)
