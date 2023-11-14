from ppo_and_friends.environments.abmarl.wrappers import AbmarlWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import EnvironmentRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.abmarl.envs.maze_env import sm_abmarl_grid_pos_maze
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AbmarlGridPosMazeRunner(EnvironmentRunner):

    def run(self):
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = 0.0005

        entropy_weight = LinearStepScheduler(
            status_key      = "longest episode",
            initial_value   = 0.03,
            compare_fn      = np.less_equal,
            status_triggers = [200, 100, 20],
            step_values     = [1e-2, 1e-3, 1e-4])

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"     : (-10., 10.),
            "entropy_weight"     : entropy_weight,
        }

        policy_name = "abmarl-maze"
        policy_mapping_fn = lambda *args : policy_name

        env_generator = lambda : \
                AbmarlWrapper(env               = sm_abmarl_grid_pos_maze,
                              policy_mapping_fn = policy_mapping_fn)

        policy_settings = { policy_name : \
            (None,
             env_generator().observation_space["navigator"],
             env_generator().critic_observation_space["navigator"],
             env_generator().action_space["navigator"],
             policy_args)
        }

        ts_per_rollout = self.get_adjusted_ts_per_rollout(256)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 128,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 256,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
