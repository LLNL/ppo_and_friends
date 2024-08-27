import gym as old_gym
import rware
import ast
from ppo_and_friends.environments.gym.wrappers import MultiAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
import ppo_and_friends.networks.actor_critic.multi_agent_transformer as mat
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.gym.version_wrappers import Gym21ToGymnasium
import torch.nn as nn
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from ppo_and_friends.policies.mat_policy import MATPolicy

from mpi4py import MPI
comm      = MPI.COMM_WORLD

@ppoaf_runner
class MATRobotWarehouseHardRunner(GymRunner):

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

        parser.add_argument("--bs_clip_min", default=-np.inf, type=float)
        parser.add_argument("--bs_clip_max", default=np.inf, type=float)

        parser.add_argument("--learning_rate", default=0.0003, type=float)

        parser.add_argument("--enable_icm", type=int, default=0)
        parser.add_argument("--icm_inverse_size", type=int, default=32)
        parser.add_argument("--icm_inverse_depth", type=int, default=2)
        parser.add_argument("--icm_forward_size", type=int, default=32)
        parser.add_argument("--icm_forward_depth", type=int, default=2)
        parser.add_argument("--icm_encoded_obs_dim", type=int, default=0)
        parser.add_argument("--icm_learning_rate", type=float, default=0.0003)
        parser.add_argument("--intr_reward_weight", type=float, default=1.0)

        parser.add_argument("--actor_hidden", type=int, default=256)
        parser.add_argument("--critic_hidden_mult", type=int, default=2)

        parser.add_argument("--max_ts_per_ep", type=int, default=32)
        parser.add_argument("--ts_per_rollout", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=1000)
        parser.add_argument("--epochs_per_iter", type=int, default=5)
        parser.add_argument("--soft_resets", type=int, default=0)

        #
        # Env options.
        #
        parser.add_argument("--num_agents", type=int, default=2)
        parser.add_argument("--grid_size", type=str, default="tiny",
            choices=["tiny", "small", "medium", "large"])
        parser.add_argument("--difficulty", type=str, default="easy",
            choices=["easy", "hard"])
        parser.add_argument("--env_max_steps", type=int, default=500)
        parser.add_argument("--env_kw_args", type=ast.literal_eval, default="{}")

        return parser

    def run(self):
        if old_gym.__version__ != '0.21.0':
            msg  = "ERROR: RobotWarehouse requires gym version 0.21. "
            msg += "It is not currently supported in gymnasium."
            rank_print(msg)
            comm.barrier()
            comm.Abort()

        env_str = f"rware-{self.cli_args.grid_size}-{self.cli_args.num_agents}ag-{self.cli_args.difficulty}-v1"
        rank_print(f"Using env string: {env_str}")
        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21ToGymnasium(
                    old_gym.make(
                        env_str,
                        max_steps = self.cli_args.env_max_steps,
                        **self.cli_args.env_kw_args)),

                critic_view   = "local",
                add_agent_ids = False)

        #
        # MAT kwargs
        #
        mat_kw_args  = {}

        #
        # MAPPO kwargs
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = self.cli_args.actor_hidden

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = self.cli_args.actor_hidden * self.cli_args.critic_hidden_mult

        if self.cli_args.policy == "mappo":
            ac_network  = FeedForwardNetwork
            policy_type = None

        elif self.cli_args.policy == "mat":
            ac_network  = mat.MATActorCritic
            policy_type = MATPolicy

        else:
            print(f"ERROR: unknown policy type {self.cli_args.policy}")
            comm.Abort()

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"]      = self.cli_args.icm_encoded_obs_dim
        icm_kw_args["inverse_hidden_depth"] = self.cli_args.icm_inverse_depth
        icm_kw_args["inverse_hidden_size"]  = self.cli_args.icm_inverse_size
        icm_kw_args["forward_hidden_depth"] = self.cli_args.icm_forward_depth
        icm_kw_args["forward_hidden_size"]  = self.cli_args.icm_forward_size

        bs_clip_min = self.cli_args.bs_clip_min
        bs_clip_max = self.cli_args.bs_clip_max

        policy_args = {\
            "ac_network"         : ac_network,

            #
            # MAT only.
            #
            "mat_kw_args"        : mat_kw_args,
            
            #
            # MAPPO only.
            #
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,

            "lr"                 : self.cli_args.learning_rate,
            "enable_icm"         : self.cli_args.enable_icm,
            "icm_kw_args"        : icm_kw_args,
            "icm_lr"             : self.cli_args.icm_learning_rate,
            "bootstrap_clip"     : (bs_clip_min, bs_clip_max),
            "intr_reward_weight" : self.cli_args.intr_reward_weight,
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "rware",
            env_generator = env_generator,
            policy_args   = policy_args,
            policy_type   = policy_type)

        #
        # This environment comes from arXiv:2006.07869v4.
        # This is a very sparse reward environment, and there are series of
        # complex actions that must occur in between rewards. Because of this,
        # using a large maximum timesteps per episode results in faster learning.
        # arXiv:2103.01955v2 suggests using smaller epoch counts for complex
        # environments and large batch sizes (single batches if possible).
        # Because of the sparse rewards, I've also increased the entropy
        # weight to incentivize exploration. We could also experiment with
        # using ICM here. I've disabled rewards and observation normalization
        # and clipping, mainly because they aren't mentioned in arXiv:2103.01955v2.
        # I've noticed that performance tends to go down a bit when these
        # normalizations are enabled.
        #
        self.run_ppo(
                env_generator      = env_generator,
                policy_settings    = policy_settings,
                policy_mapping_fn  = policy_mapping_fn,
                batch_size         = self.cli_args.batch_size,
                epochs_per_iter    = self.cli_args.epochs_per_iter,
                max_ts_per_ep      = self.cli_args.max_ts_per_ep,
                ts_per_rollout     = self.cli_args.ts_per_rollout,
                soft_resets        = bool(self.cli_args.soft_resets),
                normalize_obs      = False,
                obs_clip           = None,
                normalize_rewards  = False,
                reward_clip        = None,
                **self.kw_run_args)
