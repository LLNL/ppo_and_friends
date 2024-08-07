from ppo_and_friends.environments.abmarl.wrappers import AbmarlWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import EnvironmentRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
from ppo_and_friends.environments.abmarl.envs.maze_env import lg_abmarl_blind_maze
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner

@ppoaf_runner
class AbmarlBlindLargeMazeRunner(EnvironmentRunner):

    def run(self):
        #
        # The large maze is very finicky. I've found 2 general approaches to
        # solving it.
        #   1. Use long trajectories (I used 4096). This allows the agent to
        #      fully explore the space and reach the goal consistently enough
        #      to learn a good policy. The downside to this is that each rollout
        #      takes a while.
        #   2. Use shorter trajectories in conjunction with soft resets. This
        #      is an alternative approach to exploring the space and allows
        #      the agent to begin converging on a policy much sooner than
        #      the first approach. The main issue here is that soft resets
        #      will add more stochasticity to the trajectories and slow down
        #      the rate of convergence. We can combat this by wrapping the
        #      soft reset flag in a scheduler that disables it when it starts
        #      consistently reaching the goal.
        #
        # Aside from the above, I've also found that annealing exploration over
        # time greatly improves results.
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"]     = 0
        icm_kw_args["inverse_hidden_size"] = 128
        icm_kw_args["forward_hidden_size"] = 128

        lr = 0.0005
        #
        # Once we start consistently reaching the goal, we can stop
        # exploring so much. Two approaches are below.
        #

        #
        # This first approach takes a bit longer than the second,
        # but it's more generalizable to different MPI distributions.
        #
        rollout_length = 4096
        soft_resets    = False
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

        #
        # The approach below is faster when you have access to lots of
        # processors. I used 36 and was able to generate a converged policy
        # in under 30 minutes. The problem with this strategy is that I
        # didn't spend time generalizing the soft reset trigger.
        #
        #rollout_length = 512
        #intr_reward_weight = LinearStepScheduler(
        #    status_key      = "longest episode",
        #    initial_value   = 1e-1,
        #    compare_fn      = np.less_equal,
        #    status_triggers = [500, 256, 128, 95,],
        #    step_values     = [1e-2, 1e-4, 1e-6, 0.0,])

        #entropy_weight = LinearStepScheduler(
        #    status_key      = "longest episode",
        #    initial_value   = 0.03,
        #    compare_fn      = np.less_equal,
        #    status_triggers = [500, 256, 128, 95,],
        #    step_values     = [1e-2, 1e-3, 1e-4, 0.0])

        #soft_resets = LinearStepScheduler(
        #    status_key      = "iteration",
        #    initial_value   = True,
        #    compare_fn      = np.greater_equal,
        #    status_triggers = [50,],
        #    step_values     = [False,])

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
                AbmarlWrapper(env               = lg_abmarl_blind_maze,
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
                     ts_per_rollout     = rollout_length,
                     normalize_values   = True,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     soft_resets        = soft_resets,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
