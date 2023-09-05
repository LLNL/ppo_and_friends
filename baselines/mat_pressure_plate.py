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
from ppo_and_friends.policies.mat_policy import MATPolicy

from mpi4py import MPI
comm      = MPI.COMM_WORLD

@ppoaf_runner
class PressurePlateRunner(GymRunner):

    def run(self):
        if old_gym.__version__ != '0.21.0':
            msg  = "ERROR: PressurePlate requires gym version 0.21. "
            msg += "It is not currently supported in gymnasium."
            rank_print(msg)
            comm.barrier()
            comm.Abort()

        # NOTE: the compatibility wrapper that gym comes with wasn't
        # working here...
        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21ToGymnasium(old_gym.make('pressureplate-linear-4p-v0')),

                #critic_view   = "global",#FIXME: I think we can't use policy view for gym envs?
                critic_view   = "local",

                add_agent_ids = True)

        mat_kw_args  = {}

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 2000000,
            max_value     = 0.001,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 2000000,
            #status_max    = 1000000,
            max_value     = 0.02,
            min_value     = 0.01)

        #entropy_weight = LinearStepScheduler(
        #    status_key      = "longest run",
        #    initial_value   = 0.02,
        #    compare_fn      = np.less_equal,
        #    status_triggers = [500, 300],
        #    step_values     = [0.015, 0.01])

        policy_args = {\
            "mat_kw_args"      : mat_kw_args,
            "lr"               : lr,
            "entropy_weight"   : entropy_weight,
            "bootstrap_clip"   : (-3, 0),
            #"bootstrap_clip"   : (-100, 100),
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_type   = MATPolicy,
            policy_name   = "pressure-plate",
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key  = "longest run",
            compare_fn  = np.less_equal,
            persistent  = True)

        ts_per_rollout = self.get_adjusted_ts_per_rollout(512)

        self.run_ppo(env_generator      = env_generator,
                     save_when          = save_when,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 10000,
                     epochs_per_iter    = 15,
                     max_ts_per_ep      = 128,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     soft_resets        = True,
                     **self.kw_run_args)