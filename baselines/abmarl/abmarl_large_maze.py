from ppo_and_friends.environments.abmarl.wrappers import AbmarlWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import EnvironmentRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.abmarl.envs.maze_env import lg_abmarl_maze
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AbmarlLargeMazeRunner(EnvironmentRunner):

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
        icm_kw_args["encoded_obs_dim"] = 9

        lr = 0.0005
        soft_resets = False
        intr_reward_weight = LinearStepScheduler(
            status_key      = "longest episode",
            initial_value   = 1e-3,
            compare_fn      = np.less_equal,
            status_triggers = [4000, 3000, 2000, 500,],
            step_values     = [1e-4, 1e-5, 1e-6, 0.0,])

        entropy_weight = LinearStepScheduler(
            status_key      = "longest episode",
            initial_value   = 0.04,
            compare_fn      = np.less_equal,
            status_triggers = [4000, 3000, 2000, 500],
            step_values     = [1e-2, 1e-3, 1e-4, 0.0])

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "icm_lr"             : 0.0005,
            "lr"                 : lr,
            "bootstrap_clip"     : (-10., 10.),
            "enable_icm"         : True,
            "icm_kw_args"        : icm_kw_args,
            "intr_reward_weight" : intr_reward_weight,
            "entropy_weight"     : entropy_weight,
        }

        policy_name = "abmarl-maze"
        policy_mapping_fn = lambda *args : policy_name

        env_generator = lambda : \
                AbmarlWrapper(env               = lg_abmarl_maze,
                              policy_mapping_fn = policy_mapping_fn)

        policy_settings = { policy_name : \
            (None,
             env_generator().observation_space["navigator"],
             env_generator().critic_observation_space["navigator"],
             env_generator().action_space["navigator"],
             policy_args)
        }

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 512,
                     ts_per_rollout     = 4096,
                     normalize_values   = True,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     soft_resets        = soft_resets,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
