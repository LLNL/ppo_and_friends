import gym as old_gym
from ppo_and_friends.environments.gym.wrappers import MultiAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.gym.version_wrappers import Gym21ToGymnasium
import torch.nn as nn
from ppo_and_friends.utils.mpi_utils import rank_print
import pressureplate
from ppo_and_friends.runners.runner_tags import ppoaf_runner
from packaging import version
import ppo_and_friends.networks.actor_critic.multi_agent_transformer as mat
from ppo_and_friends.policies.mat_policy import MATPolicy

from mpi4py import MPI
comm = MPI.COMM_WORLD

@ppoaf_runner
class PressurePlateRunner(GymRunner):

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
        #
        # For study:
        #   num_agents: 4, 5, 6
        #   use_icm: True, False
        #   ts_per_rollout: 256
        #   policy: MAT, MAPPO
        #
        parser.add_argument("--num_agents", type=int, default=4)
        parser.add_argument("--use_icm", type=int, default=0)
        parser.add_argument("--ts_per_rollout", default=256, type=int)
        parser.add_argument("--policy", default='mappo', type=str, choices=["mat", "mappo"])
        parser.add_argument("--bs_clip_min", default=-np.inf, type=float)
        parser.add_argument("--bs_clip_max", default=np.inf, type=float)
        parser.add_argument("--soft_resets", type=int, default=0)
        return parser

    def run(self):
        if old_gym.__version__ != '0.21.0':
            msg  = "ERROR: PressurePlate requires gym version 0.21. "
            msg += "It is not currently supported in gymnasium."
            rank_print(msg)
            comm.Abort()

        import pyglet
        try:
            if version.parse(pyglet.version) == version.parse('1.3.2'):
                msg  = "WARNING: PressurePlate requires pyget version == 1.3.2 "
                msg += "for rendering. Rendering will fail with later versions."
                rank_print(msg)
        except:
            pass

        # NOTE: the compatibility wrapper that gym comes with wasn't
        # working here...
        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21ToGymnasium(old_gym.make(f"pressureplate-linear-{self.cli_args.num_agents}p-v0")),
                critic_view   = "local",
                add_agent_ids = True)

        if self.cli_args.policy == "mappo":
            ac_network  = FeedForwardNetwork
            policy_type = None

        elif self.cli_args.policy == "mat":
            ac_network  = mat.MATActorCritic
            policy_type = MATPolicy

        else:
            print(f"ERROR: unknown policy type {self.cli_args.policy}")
            comm.Abort()

        #
        # MAT kwargs.
        #
        mat_kw_args = {}

        #
        # MAPPO kwargs.
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 5_000_000,
            max_value     = 0.001,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 5_000_000,
            max_value     = 0.02,
            min_value     = 0.01)

        policy_args = {\
            "ac_network"       : ac_network,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : entropy_weight,
            "bootstrap_clip"   : (self.cli_args.bs_clip_min, self.cli_args.bs_clip_max),

            #
            # ICM only.
            #
            "intr_reward_weight" : 1./1000.,
            "enable_icm"         : self.cli_args.use_icm,
            "icm_lr"             : 0.0003,

            #
            # MAT only.
            #
            "mat_kw_args"      : mat_kw_args,

            #
            # MAPPO only
            #
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "pressure-plate",
            env_generator = env_generator,
            policy_args   = policy_args,
            policy_type   = policy_type)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = self.cli_args.ts_per_rollout,
                     epochs_per_iter    = 15,
                     max_ts_per_ep      = 128,
                     ts_per_rollout     = self.cli_args.ts_per_rollout,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     soft_resets        = self.cli_args.soft_resets,
                     **self.kw_run_args)
