"""
    A home for environment "launchers", defined as simple functions
    that initialize training for a specific environment.
"""
import gymnasium as gym
from abc import ABC, abstractmethod
from ppo_and_friends.ppo import PPO
from ppo_and_friends.testing import test_policy
from ppo_and_friends.networks.actor_critic_networks import FeedForwardNetwork, AtariPixelNetwork
from ppo_and_friends.networks.actor_critic_networks import SplitObsNetwork
from ppo_and_friends.networks.actor_critic_networks import LSTMNetwork
from ppo_and_friends.networks.encoders import LinearObservationEncoder
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.environments.gym_wrappers import SingleAgentGymWrapper
from ppo_and_friends.environments.gym_wrappers import MultiAgentGymWrapper
from ppo_and_friends.environments.gym_envs.multi_binary import MultiBinaryCartPoleWrapper
from ppo_and_friends.environments.gym_envs.multi_binary import MultiBinaryLunarLanderWrapper
from ppo_and_friends.environments.version_wrappers import Gym21To26
from ppo_and_friends.policies.utils import get_single_policy_defaults
from .atari_wrappers import *
import torch.nn as nn
from ppo_and_friends.utils.schedulers import *

try:
    from ppo_and_friends.environments.abmarl_wrappers import AbmarlWrapper
    from ppo_and_friends.environments.abmarl_envs.maze_env import sm_abmarl_maze
    from ppo_and_friends.environments.abmarl_envs.maze_env import sm_abmarl_blind_maze
    from ppo_and_friends.environments.abmarl_envs.maze_env import lg_abmarl_maze
    from ppo_and_friends.environments.abmarl_envs.reach_the_target import abmarl_rtt_env
    from ppo_and_friends.environments.abmarl_envs.maze_env import lg_abmarl_blind_maze
except Exception as e:
    print(e)
    msg = "WARNING: unable to import Abmarl."
    print(msg)

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def get_gym_render_mode(kw_launch_args):
    if kw_launch_args["render"]:
        return "human"
    else:
        return None

class EnvironmentLauncher(ABC):
    """
        A base class for launching RL environments.
    """

    def __init__(self,
                 **kw_launch_args):
        """
            Arguments:
                kw_launch_args    Keywoard arguments for launching training.
        """
        self.kw_launch_args = kw_launch_args

    @abstractmethod
    def launch(self):
        raise NotImplementedError

    def run_ppo(self,
                policy_settings,
                policy_mapping_fn,
                env_generator,
                device,
                test                  = False,
                explore_while_testing = False,
                num_timesteps         = 1_000_000,
                render_gif            = False,
                num_test_runs         = 1,
                **kw_args):

        """
            Run the PPO algorithm.
        """

        ppo = PPO(policy_settings   = policy_settings,
                  policy_mapping_fn = policy_mapping_fn,
                  env_generator     = env_generator,
                  device            = device,
                  test_mode         = test,
                  **kw_args)

        #
        # Pickling is a special case. It allows users to save the ppo class
        # for use elsewhere. So, we skip training if pickling is requested.
        #
        pickling = "pickle_class" in kw_args and kw_args["pickle_class"]

        if test:
            test_policy(ppo,
                        explore_while_testing,
                        render_gif,
                        num_test_runs,
                        device,
                        **kw_args)

        elif not pickling:
            ppo.learn(num_timesteps)


###############################################################################
#                            Classic Control                                  #
###############################################################################

class CartPoleLauncher(EnvironmentLauncher):

    def launch(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('CartPole-v0',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["activation"] = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = 0.0002
        ts_per_rollout = num_procs * 256

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(**self.kw_launch_args,
                     save_when          = save_when,
                     env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     ts_per_rollout     = ts_per_rollout,
                     max_ts_per_ep      = 32,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     normalize_adv      = True)


class BinaryCartPoleLauncher(EnvironmentLauncher):
    """
        This is merely the original CartPole environment wrapped in a
        MultiBinary action space. It's used to test our multi-binary
        action space support.
    """

    def launch(self):
        env_generator = lambda : SingleAgentGymWrapper(
            MultiBinaryCartPoleWrapper(gym.make('CartPole-v0',
                render_mode = get_gym_render_mode(self.kw_launch_args))))

        actor_kw_args = {}
        actor_kw_args["activation"] = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = 0.0002

        ts_per_rollout = num_procs * 256

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(**self.kw_launch_args,
                     save_when          = save_when,
                     env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     ts_per_rollout     = ts_per_rollout,
                     max_ts_per_ep      = 32,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     normalize_adv      = True)


class PendulumLauncher(EnvironmentLauncher):

    def launch(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Pendulum-v1',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 32

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = 0.0003

        ts_per_rollout = num_procs * 512

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "dynamic_bs_clip"  : True,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = 32,
                     epochs_per_iter    = 8,
                     obs_clip           = None,
                     reward_clip        = None,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     **self.kw_launch_args)


class MountainCarLauncher(EnvironmentLauncher):

    def launch(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('MountainCar-v0',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {"activation" : nn.LeakyReLU()}
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 128

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"] = 2

        lr = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = 0.0003,
            status_triggers = [35, 50, 60],
            step_values     = [0.00025, 0.0002, 0.0001])

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-10, 10),
            "enable_icm"       : True,
            "icm_kw_args"      : icm_kw_args,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

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
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     epochs_per_iter    = 32,
                     max_ts_per_ep      = 200,
                     ext_reward_weight  = 1./100.,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     normalize_values   = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_launch_args)


class MountainCarContinuousLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('MountainCarContinuous-v0',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # Extra args for the actor critic models.
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  =  nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 128

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"] = 2

        lr = 0.0003

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"     : (-10, 10),
            "enable_icm"         : True,
            "icm_kw_args"        : icm_kw_args,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        #
        # I've noticed that normalizing rewards and observations
        # can slow down learning at times. It's not by much (maybe
        # 10-50 iterations).
        #
        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = 200,
                     ext_reward_weight  = 1./100.,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     normalize_values   = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_launch_args)


class AcrobotLauncher(EnvironmentLauncher):

    def launch(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Acrobot-v1',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = {}
        critic_kw_args["hidden_size"] = 128

        lr = LinearScheduler(
            status_key = "iteration",
            status_max = 2000,
            max_value  = 0.0003,
            min_value  = 0.0)

        ts_per_rollout = num_procs * 512

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-10., 10.),
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     **self.kw_launch_args)


###############################################################################
#                                Box 2D                                       #
###############################################################################


class LunarLanderLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('LunarLander-v2',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # Extra args for the actor critic models.
        # I find that leaky relu does much better with the lunar
        # lander env.
        #
        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 1000000,
            max_value     = 0.0005,
            min_value     = 0.0001)

        #
        # Running with 2 processors works well here.
        #
        ts_per_rollout = num_procs * 1024

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(env_generator       = env_generator,
                     save_when           = save_when,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     max_ts_per_ep       = 128,
                     ts_per_rollout      = ts_per_rollout,
                     batch_size          = 512,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_launch_args)


class BinaryLunarLanderLauncher(EnvironmentLauncher):
    """
        This is merely the original LunarLander environment wrapped in a
        MultiBinary action space. It's used to test our multi-binary
        action space support.
    """

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(
                MultiBinaryLunarLanderWrapper(gym.make('LunarLander-v2',
                render_mode = get_gym_render_mode(self.kw_launch_args))))

        #
        # Extra args for the actor critic models.
        # I find that leaky relu does much better with the lunar
        # lander env.
        #
        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 200,
            max_value     = 0.0003,
            min_value     = 0.0001)

        #
        # Running with 2 processors works well here.
        #
        ts_per_rollout = num_procs * 1024

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(env_generator       = env_generator,
                     save_when           = save_when,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     max_ts_per_ep       = 128,
                     ts_per_rollout      = ts_per_rollout,
                     batch_size          = 512,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_launch_args)


class LunarLanderContinuousLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('LunarLanderContinuous-v2',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # Lunar lander observations are organized as follows:
        #    Positions: 2
        #    Positional velocities: 2
        #    Angle: 1
        #    Angular velocities: 1
        #    Leg contact: 2
        #
        actor_kw_args = {}

        #
        # Extra args for the actor critic models.
        # I find that leaky relu does much better with the lunar
        # lander env.
        #
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 100,
            max_value     = 0.0003,
            min_value     = 0.0001)

        #
        # Running with 2 processors works well here.
        #
        ts_per_rollout = num_procs * 1024

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-10., 10.),
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key     = "extrinsic score avg",
            status_preface = "single_agent",
            compare_fn     = np.greater_equal,
            persistent     = True)

        self.run_ppo(env_generator       = env_generator,
                     save_when           = save_when,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     max_ts_per_ep       = 32,
                     ts_per_rollout      = ts_per_rollout,
                     epochs_per_iter     = 16,
                     batch_size          = 512,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_launch_args)


class BipedalWalkerLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('BipedalWalker-v3',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # The lidar observations are the last 10.
        #
        actor_kw_args = {}

        #
        # I've found that a lower std offset greatly improves performance
        # stability in this environment. Also, most papers suggest that using Tanh
        # provides the best performance, but I find that LeakyReLU works better
        # here.
        #
        actor_kw_args["std_offset"] = 0.1
        actor_kw_args["activation"] = nn.LeakyReLU()

        #
        # You can also use an LSTM or Split Observation network here,
        # but I've found that a simple MLP learns faster both in terms
        # of iterations and wall-clock time. The LSTM is the slowest
        # of the three options, which I would assume is related to the
        # fact that velocity information is already contained in the
        # observations, but it's a bit surprising that we can't infer
        # extra "history" information from the lidar.
        #
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 200,
            max_value     = 0.0003,
            min_value     = 0.0001)

        ts_per_rollout = num_procs * 512

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-1., 10.),
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        #
        # Thresholding the reward to a low of -1 doesn't drastically
        # change learning, but it does help a bit. Clipping the bootstrap
        # reward to the same range seems to help with stability.
        #
        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     batch_size          = 512,
                     max_ts_per_ep       = 32,
                     ts_per_rollout      = ts_per_rollout,
                     normalize_adv       = True,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-1., 10.),
                     **self.kw_launch_args)


class BipedalWalkerHardcoreLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('BipedalWalkerHardcore-v3',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["std_offset"]  = 0.1
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        #
        # This environment is a pretty challenging one and can be
        # very finicky. Learning rate and reward clipping have a
        # pretty powerfull impact on results, and it can be tricky
        # to get these right. Here's what I've found works best:
        #
        #   1. First, run the problem with a pretty standard learning
        #      rate (0.0001 works well), and use a conservative reward
        #      clipping of (-1, 10). Clipping the reward at -1 results
        #      in the agent learning a good gait pretty early on.
        #
        #   2. After a while (roughly 7000->8000 iterations), the agent should
        #      have a pretty solid policy. Running tests will show that
        #      it can regularly reach scores over 300, but averaging
        #      over 100 runs will likely be in the 200s. My impression
        #      here is that the -1 reward clipping, while allowing the
        #      agent to learn a good gait quickly, also causes the agent
        #      be less concerned with failures. So, at this point, I find
        #      that adjusting the lower bound of the clip to the standard
        #      -10 value allows the agent to learn that falling is
        #      actually really bad. I also lower the learning rate here
        #      to help with stability in this last phase. This last bit
        #      of learning can take a while (~10,000 -> 11,000 iterations).
        #
        # The above is all automated with the settings used below. I typically
        # run with 4 processors. The resulting policy can regularly reach average
        # scores of 320+ over 100 test runs.
        #
        lr = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = 0.0001,
            status_triggers = [3900,],
            step_values     = [0.00001,])

        reward_clip_min = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = -1.,
            status_triggers = [4000,],
            step_values     = [-10.,])

        bs_clip_min = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = -1.,
            status_triggers = [4000,],
            step_values     = [-10.,])

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (bs_clip_min, 10.),
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_obs      = True,
                     normalize_rewards  = True,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (reward_clip_min, 10.),
                     **self.kw_launch_args)


###############################################################################
#                                Atari                                        #
###############################################################################


#FIXME: depricated now!
class BreakoutPixelsLauncher(EnvironmentLauncher):

    def launch(self):
        if self.kw_launch_args["render"]:
            #
            # NOTE: we don't want to explicitly call render for atari games.
            # They have more advanced ways of rendering.
            #
            self.kw_launch_args["render"] = False

            gym_generator = lambda : gym.make(
                    'Breakout-v4',
                    repeat_action_probability = 0.0,
                    frameskip                 = 1,
                    apply_api_compatibility   = True,
                    render_mode               = 'human')
        else:
            gym_generator = lambda : gym.make(
                    'Breakout-v0',
                    repeat_action_probability = 0.0,
                    apply_api_compatibility   = True,
                    frameskip                 = 1)

        breakout_generator = lambda : BreakoutPixelsEnvWrapper(
            env              = gym_generator(),
            allow_life_loss  = self.kw_launch_args["test"],
            hist_size        = 4,
            skip_k_frames    = 4)

        env_generator = lambda : \
            SingleAgentGymWrapper(breakout_generator())

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        critic_kw_args = actor_kw_args.copy()

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 4000,
            max_value     = 0.0003,
            min_value     = 0.0)

        policy_args = {\
            "ac_network"       : AtariPixelNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-1., 1.),
            "target_kl"        : 0.2,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     ts_per_rollout     = ts_per_rollout,
                     max_ts_per_ep      = 64,
                     epochs_per_iter    = 30,
                     reward_clip        = (-1., 1.),
                     **self.kw_launch_args)


class BreakoutRAMLauncher(EnvironmentLauncher):

    def launch(self):
        if self.kw_launch_args["render"]:
            #
            # NOTE: we don't want to explicitly call render for atari games.
            # They have more advanced ways of rendering.
            #
            self.kw_launch_args["render"] = False

            gym_generator = lambda : gym.make(
                'Breakout-ram-v0',
                repeat_action_probability = 0.0,
                frameskip = 1,
                render_mode = 'human')
        else:
            gym_generator = lambda : gym.make(
                'Breakout-ram-v0',
                repeat_action_probability = 0.0,
                frameskip = 1)

        breakout_generator = lambda : BreakoutRAMEnvWrapper(
            env              = gym_generator(),
            allow_life_loss  = self.kw_launch_args["test"],
            hist_size        = 4,
            skip_k_frames    = 4)

        env_generator = lambda : SingleAgentGymWrapper(breakout_generator())

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 4000,
            max_value     = 0.0003,
            min_value     = 0.0)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-1., 1.),
            "target_kl"        : 0.2,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     ts_per_rollout     = ts_per_rollout,
                     max_ts_per_ep      = 64,
                     epochs_per_iter    = 30,
                     reward_clip        = (-1., 1.),
                     **self.kw_launch_args)


###############################################################################
#                                MuJoCo                                       #
###############################################################################


class InvertedPendulumLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('InvertedPendulum-v4',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        policy_args = {\
            "ac_network" : FeedForwardNetwork,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     ts_per_rollout     = ts_per_rollout,
                     **self.kw_launch_args)


class InvertedDoublePendulumLauncher(EnvironmentLauncher):

    def launch(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('InvertedDoublePendulum-v4',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # Pendulum observations are organized as follows:
        #    Positions: 1
        #    Angles: 4
        #    Velocities: 3
        #    Contact forces: 3
        #
        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 128

        lr = 0.0001

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-10., 10.),
            "entropy_weight"   : 0.0,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 16,
                     ts_per_rollout     = ts_per_rollout,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     **self.kw_launch_args)

class AntLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Ant-v4',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # Ant observations are organized as follows:
        #    Positions: 13
        #    Velocities: 14
        #    Contact forces: 84
        #
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.Tanh()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 100,
            max_value     = 0.00025,
            min_value     = 0.0001)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "bootstrap_clip"   : (-10., 10.),
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 64,
                     epochs_per_iter    = 8,
                     ts_per_rollout     = ts_per_rollout,
                     obs_clip           = (-30., 30.),
                     reward_clip        = (-10., 10.),
                     **self.kw_launch_args)


class HumanoidLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Humanoid-v4',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        # Humanoid observations are a bit mysterious. See
        # https://github.com/openai/gym/issues/585
        # Here's a best guess:
        #
        #    Positions: 22
        #    Velocities: 23
        #    Center of mass based on inertia (?): 140
        #    Center of mass based on velocity (?): 84
        #    Actuator forces (?): 23
        #    Contact forces: 84
        #
        # UPDATE: more complete information on the observations cane be found
        # here:
        # https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup.py
        #
        actor_kw_args = {}

        # TODO: the current settings work pretty well, but it
        # takes a while to train. Can we do better? Some things
        # that need more exploring:
        #    std offset: is the default optimal?
        #    activation: How does leaky relu do?
        #    target_kl: we could experiment more with this.
        #    obs_clip: this seems to negatively impact results. Does that hold?
        #    entropy: we could allow entropy reg, but I'm guessing it won't help
        #             too much.
        #
        actor_kw_args["activation"]       = nn.Tanh()
        actor_kw_args["distribution_min"] = -0.4
        actor_kw_args["distribution_max"] = 0.4
        actor_kw_args["hidden_size"]      = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = 0.0001

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     epochs_per_iter    = 8,
                     max_ts_per_ep      = 16,
                     ts_per_rollout     = ts_per_rollout,
                     reward_clip        = (-10., 10.),
                     **self.kw_launch_args)


class HumanoidStandUpLauncher(EnvironmentLauncher):

    def launch(self):
        #
        # NOTE: this is an UNSOVLED environment.
        #
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('HumanoidStandup-v4',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        #
        #    Positions: 22
        #    Velocities: 23
        #    Center of mass based on inertia (?): 140
        #    Center of mass based on velocity (?): 84
        #    Actuator forces (?): 23
        #    Contact forces: 84
        #
        # UPDATE: more complete information on the observations cane be found
        # here:
        # https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup.py
        #
        actor_kw_args = {}
        actor_kw_args["activation"]       = nn.Tanh()
        actor_kw_args["distribution_min"] = -0.4
        actor_kw_args["distribution_max"] = 0.4
        actor_kw_args["hidden_size"]      = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 200,
            max_value     = 0.0003,
            min_value     = 0.0001)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 32,
                     epochs_per_iter    = 8,
                     ts_per_rollout     = ts_per_rollout,
                     obs_clip           = None,
                     reward_clip        = (-10., 10.),
                     **self.kw_launch_args)


#FIXME: busted?
class Walker2DLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Walker2d-v4',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.Tanh()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 600,
            max_value     = 0.0003,
            min_value     = 0.0001)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : 0.0,
            "target_kl"        : 0.015,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 1024

        #
        # arXiv:2006.05990v1 suggests that value normalization significantly hurts
        # performance in walker2d. I also find this to be the case.
        #
        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     max_ts_per_ep      = 16,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = False,
                     obs_clip           = (-10., 10.),
                     reward_clip        = (-10., 10.),
                     **self.kw_launch_args)

#FIXME: busted?
class HopperLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Hopper-v3',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.Tanh()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearStepScheduler(
            status_key      = "iteration",
            initial_value   = 0.0003,
            status_triggers = [400,],
            step_values     = [0.0001,])

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : 0.0,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 1024

        #
        # I find that value normalization hurts the hopper environment training.
        # That may be a result of it's combination with other settings in here.
        #
        self.run_ppo(
                env_generator      = env_generator,
                policy_settings    = policy_settings,
                policy_mapping_fn  = policy_mapping_fn,
                batch_size         = 512,
                max_ts_per_ep      = 16,
                epochs_per_iter    = 8,
                ts_per_rollout     = ts_per_rollout,
                normalize_values   = False,
                obs_clip           = (-10., 10.),
                reward_clip        = (-10., 10.),
                **self.kw_launch_args)


class HalfCheetahLauncher(EnvironmentLauncher):

    def launch(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('HalfCheetah-v3',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = 0.0001
        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        #
        # Normalizing values seems to stabilize results in this env.
        #
        self.run_ppo(
                env_generator      = env_generator,
                policy_settings    = policy_settings,
                policy_mapping_fn  = policy_mapping_fn,
                batch_size         = 512,
                max_ts_per_ep      = 32,
                ts_per_rollout     = ts_per_rollout,
                obs_clip           = (-10., 10.),
                reward_clip        = (-10., 10.),
                **self.kw_launch_args)


class SwimmerLauncher(EnvironmentLauncher):

    def launch(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(gym.make('Swimmer-v3',
                render_mode = get_gym_render_mode(self.kw_launch_args)))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = 0.0001

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 1024

        self.run_ppo(
                env_generator      = env_generator,
                policy_settings    = policy_settings,
                policy_mapping_fn  = policy_mapping_fn,
                batch_size         = 512,
                max_ts_per_ep      = 32,
                ts_per_rollout     = ts_per_rollout,
                obs_clip           = (-10., 10.),
                reward_clip        = (-10., 10.),
                **self.kw_launch_args)


###############################################################################
#                          Multi-Agent Gym                                    #
###############################################################################

class RobotWarehouseTinyLauncher(EnvironmentLauncher):

    def launch(self):
        import rware

        env_generator = lambda : \
            MultiAgentGymWrapper(
                gym.make('rware-tiny-3ag-v1'),
                critic_view = "policy",
                policy_mapping_fn = lambda *args : "rware",
                add_agent_ids = True)

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 1000000,
            max_value     = 0.0003,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 1000000,
            max_value     = 0.015,
            min_value     = 0.01)

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"     : (0., 10.),
            "entropy_weight"     : entropy_weight,
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "rware",
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 1024

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
                batch_size         = 10000,
                epochs_per_iter    = 5,
                max_ts_per_ep      = 32,
                ts_per_rollout     = ts_per_rollout,
                normalize_obs      = False,
                obs_clip           = None,
                normalize_rewards  = False,
                reward_clip        = None,
                **self.kw_launch_args)


class RobotWarehouseSmallLauncher(EnvironmentLauncher):

    def launch(self):
        import rware

        env_generator = lambda : \
            MultiAgentGymWrapper(
                gym.make('rware-small-4ag-v1'),
                add_agent_ids = True)

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 512

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 6000,
            max_value     = 0.0003,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "iteration",
            status_max    = 6000,
            max_value     = 0.05,
            min_value     = 0.01)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : entropy_weight,
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "rware",
            env_generator = env_generator,
            policy_args   = policy_args)

        ts_per_rollout = num_procs * 512

        #
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
        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 10000,
                     soft_resets        = True,
                     epochs_per_iter    = 5,
                     max_ts_per_ep      = 512,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     **self.kw_launch_args)


class PressurePlateLauncher(EnvironmentLauncher):

    def launch(self):

        try:
            import pressureplate
            import gym as old_gym
        except:
            msg  = "ERROR: unable to import the pressureplate environment. "
            msg += "This environment is installed from its git repository."
            rank_print(msg)
            comm.Abort()

        #FIXME: we have the apply_api_compatibility flag instead.
        # NOTE: the compatibility wrapper that gym comes with wasn't
        # working here...
        env_generator = lambda : \
            MultiAgentGymWrapper(
                Gym21To26(old_gym.make('pressureplate-linear-4p-v0')))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 400000,
            max_value     = 0.001,
            min_value     = 0.0001)

        entropy_weight = LinearScheduler(
            status_key    = "timesteps",
            status_max    = 400000,
            max_value     = 0.02,
            min_value     = 0.0)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            "entropy_weight"   : entropy_weight,
            "bootstrap_clip"   : (-3, 0),
        }

        #
        # All agents use the same policy, so we can use the basics here.
        #
        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            policy_name   = "pressure-plate",
            env_generator = env_generator,
            policy_args   = policy_args)

        save_when = ChangeInStateScheduler(
            status_key  = "longest run",
            compare_fn  = np.less_equal,
            persistent  = True)

        ts_per_rollout = num_procs * 512

        self.run_ppo(env_generator      = env_generator,
                     save_when          = save_when,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 10000,
                     epochs_per_iter    = 5,
                     max_ts_per_ep      = 128,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     obs_clip           = None,
                     normalize_rewards  = False,
                     reward_clip        = None,
                     soft_resets        = True,
                     **self.kw_launch_args)


###############################################################################
#                               Abmarl                                        #
###############################################################################

class AbmarlMazeLauncher(EnvironmentLauncher):

    def launch(self):
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"] = 9

        lr = 0.0005
        #
        # Once we start consistently reaching the goal, we can stop
        # exploring so much.
        #
        intr_reward_weight = LinearStepScheduler(
            status_key      = "longest run",
            initial_value   = 1e-2,
            compare_fn      = np.less_equal,
            status_triggers = [200, 100, 20,],
            step_values     = [1e-3, 1e-4, 0.0,])

        entropy_weight = LinearStepScheduler(
            status_key      = "longest run",
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
            "enable_icm"         : True,
            "icm_kw_args"        : icm_kw_args,
            "intr_reward_weight" : intr_reward_weight,
            "entropy_weight"     : entropy_weight,
        }

        policy_name = "abmarl-maze"
        policy_mapping_fn = lambda *args : policy_name

        env_generator = lambda : \
                AbmarlWrapper(env               = sm_abmarl_maze,
                              policy_mapping_fn = policy_mapping_fn)

        policy_settings = { policy_name : \
            (None,
             env_generator().observation_space["navigator"],
             env_generator().critic_observation_space["navigator"],
             env_generator().action_space["navigator"],
             policy_args)
        }

        ts_per_rollout = num_procs * 256

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
                     **self.kw_launch_args)


class AbmarlBlindMazeLauncher(EnvironmentLauncher):

    def launch(self):
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 32

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 64

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"] = 0

        lr = 0.0005

        #
        # Once we start consistently reaching the goal, we can stop
        # exploring so much.
        #
        intr_reward_weight = LinearStepScheduler(
            status_key      = "longest run",
            initial_value   = 1e-2,
            compare_fn      = np.less_equal,
            status_triggers = [200, 100, 20,],
            step_values     = [1e-3, 1e-4, 0.0,])

        entropy_weight = LinearStepScheduler(
            status_key      = "longest run",
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
            "enable_icm"         : True,
            "icm_kw_args"        : icm_kw_args,
            "entropy_weight"     : entropy_weight,
            "intr_reward_weight" : intr_reward_weight,
        }

        policy_name = "abmarl-maze"
        policy_mapping_fn = lambda *args : policy_name

        env_generator = lambda : \
                AbmarlWrapper(env               = sm_abmarl_blind_maze,
                              policy_mapping_fn = policy_mapping_fn)

        policy_settings = { policy_name : \
            (None,
             env_generator().observation_space["navigator"],
             env_generator().critic_observation_space["navigator"],
             env_generator().action_space["navigator"],
             policy_args)
        }

        ts_per_rollout = num_procs * 256

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
                     **self.kw_launch_args)


class AbmarlLargeMazeLauncher(EnvironmentLauncher):

    def launch(self):
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
            status_key      = "longest run",
            initial_value   = 1e-3,
            compare_fn      = np.less_equal,
            status_triggers = [4000, 3000, 2000, 500,],
            step_values     = [1e-4, 1e-5, 1e-6, 0.0,])

        entropy_weight = LinearStepScheduler(
            status_key      = "longest run",
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

        ts_per_rollout = num_procs * 4096

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 512,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     soft_resets        = soft_resets,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_launch_args)


class AbmarlBlindLargeMazeLauncher(EnvironmentLauncher):

    def launch(self):
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
            status_key      = "longest run",
            initial_value   = 1e-3,
            compare_fn      = np.less_equal,
            status_triggers = [4000, 3000, 2000, 500,],
            step_values     = [1e-4, 1e-5, 1e-6, 0.0,])

        entropy_weight = LinearStepScheduler(
            status_key      = "longest run",
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
        #    status_key      = "longest run",
        #    initial_value   = 1e-1,
        #    compare_fn      = np.less_equal,
        #    status_triggers = [500, 256, 128, 95,],
        #    step_values     = [1e-2, 1e-4, 1e-6, 0.0,])

        #entropy_weight = LinearStepScheduler(
        #    status_key      = "longest run",
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

        ts_per_rollout = num_procs * rollout_length

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 512,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 512,
                     ts_per_rollout     = ts_per_rollout,
                     normalize_values   = True,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     soft_resets        = soft_resets,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_launch_args)


class AbmarlReachTheTargetLauncher(EnvironmentLauncher):

    def launch(self):
        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 64

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 128

        lr = LinearScheduler(
            status_key     = "iteration",
            status_max     = 100,
            max_value      = 0.0003,
            min_value      = 0.0001)

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : lr,
            "bootstrap_clip"     : (-10, 10),
        }

        def policy_mapping_fn(agent_id):
            return 'runner' if agent_id.startswith('runner') else 'target'

        env_generator = lambda : \
                AbmarlWrapper(env               = abmarl_rtt_env,
                              policy_mapping_fn = policy_mapping_fn)

        #
        # This environment is multi-agent and requires different policies.
        #
        policy_settings = { "target" : \
            (None,
             env_generator().observation_space["target"],
             env_generator().critic_observation_space["target"],
             env_generator().action_space["target"],
             policy_args),

            "runner" :
            (None,
             env_generator().observation_space["runner0"],
             env_generator().critic_observation_space["runner0"],
             env_generator().action_space["runner0"],
             policy_args)
        }

        ts_per_rollout = num_procs * 256

        self.run_ppo(env_generator      = env_generator,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     batch_size         = 256,
                     epochs_per_iter    = 20,
                     max_ts_per_ep      = 32,
                     ts_per_rollout     = ts_per_rollout,
                     **self.kw_launch_args)
