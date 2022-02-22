"""
    A home for environment "launchers", defined as simple functions
    that initialize training for a specific environment.
"""
import gym
from ppo_and_friends.ppo import PPO
from ppo_and_friends.testing import test_policy
from ppo_and_friends.networks.actor_critic_networks import SimpleFeedForward, AtariPixelNetwork
from ppo_and_friends.networks.actor_critic_networks import SimpleSplitObsNetwork
from ppo_and_friends.networks.icm import ICM
from ppo_and_friends.networks.encoders import LinearObservationEncoder
from .gym_wrappers import *
import torch.nn as nn
from ppo_and_friends.utils.decrementers import *


def run_ppo(env,
            ac_network,
            device,
            icm_network         = ICM,
            batch_size          = 256,
            ts_per_rollout      = 1024,
            epochs_per_iter     = 10,
            target_kl           = 100.,
            lr                  = 3e-4,
            min_lr              = 1e-4,
            lr_dec              = None,
            max_ts_per_ep       = 200,
            use_gae             = True,
            use_icm             = False,
            save_best_only      = False,
            icm_beta            = 0.8,
            ext_reward_weight   = 1.0,
            intr_reward_weight  = 1.0,
            entropy_weight      = 0.01,
            actor_kw_args       = {},
            critic_kw_args      = {},
            icm_kw_args         = {},
            surr_clip           = 0.2,
            bootstrap_clip      = (-10.0, 10.0),
            dynamic_bs_clip     = False,
            gradient_clip       = 0.5,
            mean_window_size    = 100,
            normalize_adv       = True,
            normalize_obs       = False,
            normalize_rewards   = False,
            obs_clip            = None,
            reward_clip         = None,
            render              = False,
            load_state          = False,
            state_path          = "./",
            num_timesteps       = 1,
            test                = False,
            num_test_runs       = 1):

    ppo = PPO(env                = env,
              ac_network         = ac_network,
              icm_network        = icm_network,
              device             = device,
              batch_size         = batch_size,
              ts_per_rollout     = ts_per_rollout,
              lr                 = lr,
              target_kl          = target_kl,
              min_lr             = min_lr,
              lr_dec             = lr_dec,
              max_ts_per_ep      = max_ts_per_ep,
              use_gae            = use_gae,
              use_icm            = use_icm,
              save_best_only     = save_best_only,
              ext_reward_weight  = ext_reward_weight,
              intr_reward_weight = intr_reward_weight,
              entropy_weight     = entropy_weight,
              icm_kw_args        = icm_kw_args,
              actor_kw_args      = actor_kw_args,
              critic_kw_args     = critic_kw_args,
              surr_clip          = surr_clip,
              bootstrap_clip     = bootstrap_clip,
              dynamic_bs_clip    = dynamic_bs_clip,
              gradient_clip      = gradient_clip,
              normalize_adv      = normalize_adv,
              normalize_obs      = normalize_obs,
              normalize_rewards  = normalize_rewards,
              obs_clip           = obs_clip,
              reward_clip        = reward_clip,
              mean_window_size   = mean_window_size,
              render             = render,
              load_state         = load_state,
              state_path         = state_path,
              test_mode          = test)

    if test:
        test_policy(ppo,
                    num_test_runs,
                    device)
    else: 
        ppo.learn(num_timesteps)


###############################################################################
#                            Classic Control                                  #
###############################################################################

def cartpole_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False,
                 num_test_runs = 1):

    env = gym.make('CartPole-v0')

    #lr     = 0.0003
    #min_lr = 0.0001

    lr     = 0.0002
    min_lr = 0.0002

    lr_dec = LinearDecrementer(
        #max_iteration  = 200,
        max_iteration  = 1,
        max_value      = lr,
        min_value      = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 32,

            use_gae            = False,
            normalize_obs      = False,
            normalize_rewards  = False,
            normalize_adv      = True,
            #gradient_clip      = 1000000.,

            obs_clip           = (-10., 10.),
            reward_clip        = (-10., 10.),

            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            lr                 = lr,
            min_lr             = min_lr,
            lr_dec             = lr_dec,
            test               = test,
            num_test_runs      = num_test_runs)


def pendulum_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False,
                 num_test_runs = 1):

    env = gym.make('Pendulum-v1')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 32

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 1000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            max_ts_per_ep      = 32,
            use_gae            = True,
            normalize_obs      = True,
            normalize_rewards  = True,
            dynamic_bs_clip    = True,
            obs_clip           = (-10., 10.),
            reward_clip        = (-10., 10.),
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            lr                 = lr,
            min_lr             = min_lr,
            lr_dec             = lr_dec,
            test               = test,
            num_test_runs      = num_test_runs)


def mountain_car_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False,
                     num_test_runs = 1):

    env = gym.make('MountainCar-v0')

    ac_kw_args = {"activation" : nn.LeakyReLU()}
    ac_kw_args["hidden_size"] = 64

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 1000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 64,
            ts_per_rollout     = 2048,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            use_icm            = True,
            use_gae            = True,
            normalize_obs      = False,
            normalize_rewards  = False,
            obs_clip           = None,
            reward_clip        = None,
            bootstrap_clip     = (-10, 10),
            ext_reward_weight  = 1./100.,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


def mountain_car_continuous_ppo(state_path,
                                load_state,
                                render,
                                num_timesteps,
                                device,
                                test = False,
                                num_test_runs = 1):

    env = gym.make('MountainCarContinuous-v0')

    #
    # Extra args for the actor critic models.
    #
    actor_kw_args = {}
    actor_kw_args["activation"]  =  nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 8000,
        max_value     = lr,
        min_value     = min_lr)

    #
    # I've noticed that normalizing rewards and observations
    # can slow down learning at times. It's not by much (maybe
    # 10-50 iterations).
    #
    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 64,
            batch_size         = 512,
            ts_per_rollout     = 2048,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            use_icm            = True,
            use_gae            = True,
            normalize_obs      = False,
            normalize_rewards  = False,
            obs_clip           = None,
            reward_clip        = None,
            normalize_adv      = True,
            bootstrap_clip     = (-10., 10.),
            ext_reward_weight  = 1./100.,
            intr_reward_weight = 50.,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


def acrobot_ppo(state_path,
                load_state,
                render,
                num_timesteps,
                device,
                test = False,
                num_test_runs = 1):

    env = gym.make('Acrobot-v1')

    actor_kw_args = {}
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = {}
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 2000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 32,
            ts_per_rollout     = 2048,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            use_gae            = True,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            normalize_obs      = True,
            normalize_rewards  = True,
            obs_clip           = (-10., 10.),
            reward_clip        = (-10., 10.),
            bootstrap_clip     = (-10., 10.),
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


###############################################################################
#                                Box 2D                                       #
###############################################################################

def lunar_lander_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False,
                     num_test_runs = 1):

    env = gym.make('LunarLander-v2')

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

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 1000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            max_ts_per_ep       = 128,
            ts_per_rollout      = 2048,
            batch_size          = 512,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            dynamic_bs_clip     = False,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            target_kl           = 0.015,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            test                = test,
            num_test_runs       = num_test_runs)


def lunar_lander_continuous_ppo(state_path,
                                load_state,
                                render,
                                num_timesteps,
                                device,
                                test = False,
                                num_test_runs = 1):

    env = gym.make('LunarLanderContinuous-v2')

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

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 500,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            max_ts_per_ep       = 32,
            ts_per_rollout      = 2048,
            batch_size          = 512,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            dynamic_bs_clip     = False,
            target_kl           = 0.015,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            test                = test,
            num_test_runs       = num_test_runs)


def bipedal_walker_ppo(state_path,
                       load_state,
                       render,
                       num_timesteps,
                       device,
                       test = False,
                       num_test_runs = 1):

    env = gym.make('BipedalWalker-v3')

    #
    # The lidar observations are the last 10.
    #
    actor_kw_args = {}
    actor_kw_args["split_start"]    = env.observation_space.shape[0] - 10
    actor_kw_args["hidden_left"]    = 64
    actor_kw_args["hidden_right"]   = 64

    #
    # I've found that a lower std offset greatly improves performance
    # in this environment. Also, most papers suggest that using Tanh
    # provides the best performance, but I find that ReLU works better
    # here, which is the default.
    #
    actor_kw_args["std_offset"] = 0.1

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_left"]  = 128
    critic_kw_args["hidden_right"] = 128

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 2000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = 2048,
            use_gae             = True,
            target_kl           = 0.3,
            save_best_only      = False,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


###############################################################################
#                                Atari                                        #
###############################################################################


def assault_ram_ppo(state_path,
                    load_state,
                    render,
                    num_timesteps,
                    device,
                    test = False,
                    num_test_runs = 1):

    if render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env = gym.make(
            'Assault-ram-v0',
            repeat_action_probability = 0.0,
            frameskip = 1,
            render_mode='human')
    else:
        env = gym.make(
            'Assault-ram-v0',
            repeat_action_probability = 0.0,
            frameskip = 1)

    wrapped_env = AssaultRAMEnvWrapper(
        env              = env,
        allow_life_loss  = test,
        hist_size        = 4,
        skip_k_frames    = 4)

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = wrapped_env,
            ac_network         = SimpleFeedForward,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            batch_size         = 512,
            ts_per_rollout     = 2048,
            max_ts_per_ep      = 64,
            use_gae            = True,
            epochs_per_iter    = 30,
            reward_clip        = (-1., 1.),
            bootstrap_clip     = (-1., 1.),
            target_kl          = 0.2,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


def assault_pixels_ppo(state_path,
                       load_state,
                       render,
                       num_timesteps,
                       device,
                       test = False,
                       num_test_runs = 1):

    #
    # NOTE: there appears to be something wrong with this game's controls.
    # The "UP" action appears to be a NOOP. This limits how far you can
    # actually get in the game.
    #

    if render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env = gym.make(
            'Assault-v0',
            repeat_action_probability = 0.0,
            frameskip = 1,
            render_mode='human')
    else:
        env = gym.make(
            'Assault-v0',
            repeat_action_probability = 0.0,
            frameskip = 1)

    wrapped_env = AssaultPixelsEnvWrapper(
        env             = env,
        allow_life_loss = test,
        hist_size       = 4,
        skip_k_frames   = 4)

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    critic_kw_args = actor_kw_args.copy()

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                  = wrapped_env,
            ac_network           = AtariPixelNetwork,
            actor_kw_args        = actor_kw_args,
            critic_kw_args       = critic_kw_args,
            batch_size           = 512,
            ts_per_rollout       = 2048,
            max_ts_per_ep        = 64,
            epochs_per_iter      = 30,
            reward_clip          = (-1., 1.),
            bootstrap_clip       = (-1., 1.),
            target_kl            = 0.2,
            lr_dec               = lr_dec,
            lr                   = lr,
            min_lr               = min_lr,
            use_gae              = True,
            state_path           = state_path,
            load_state           = load_state,
            render               = render,
            num_timesteps        = num_timesteps,
            device               = device,
            test                 = test,
            num_test_runs        = num_test_runs)


def breakout_pixels_ppo(state_path,
                        load_state,
                        render,
                        num_timesteps,
                        device,
                        test = False,
                        num_test_runs = 1):

    if render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env = gym.make(
            'Breakout-v0',
            repeat_action_probability = 0.0,
            frameskip = 1,
            render_mode = 'human')
    else:
        env = gym.make(
            'Breakout-v0',
            repeat_action_probability = 0.0,
            frameskip = 1)

    wrapped_env = BreakoutPixelsEnvWrapper(
        env              = env,
        allow_life_loss  = test,
        hist_size        = 4,
        skip_k_frames    = 4)

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    critic_kw_args = actor_kw_args.copy()

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                  = wrapped_env,
            ac_network           = AtariPixelNetwork,
            actor_kw_args        = actor_kw_args,
            critic_kw_args       = critic_kw_args,
            batch_size           = 512,
            ts_per_rollout       = 2048,
            max_ts_per_ep        = 64,
            epochs_per_iter      = 30,
            reward_clip          = (-1., 1.),
            bootstrap_clip       = (-1., 1.),
            target_kl            = 0.2,
            lr_dec               = lr_dec,
            lr                   = lr,
            min_lr               = min_lr,
            use_gae              = True,
            state_path           = state_path,
            load_state           = load_state,
            render               = render,
            num_timesteps        = num_timesteps,
            device               = device,
            test                 = test,
            num_test_runs        = num_test_runs)


def breakout_ram_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False,
                     num_test_runs = 1):

    if render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env = gym.make(
            'Breakout-ram-v0',
            repeat_action_probability = 0.0,
            frameskip = 1,
            render_mode = 'human')
    else:
        env = gym.make(
            'Breakout-ram-v0',
            repeat_action_probability = 0.0,
            frameskip = 1)

    wrapped_env = BreakoutRAMEnvWrapper(
        env              = env,
        allow_life_loss  = test,
        hist_size        = 4,
        skip_k_frames    = 4)

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = wrapped_env,
            ac_network         = SimpleFeedForward,
            actor_kw_args      = actor_kw_args,
            critic_kw_args     = critic_kw_args,
            batch_size         = 512,
            ts_per_rollout     = 2048,
            max_ts_per_ep      = 64,
            use_gae            = True,
            epochs_per_iter    = 30,
            reward_clip        = (-1., 1.),
            bootstrap_clip     = (-1., 1.),
            target_kl          = 0.2,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


###############################################################################
#                                MuJoCo                                       #
###############################################################################


def inverted_pendulum_ppo(state_path,
                          load_state,
                          render,
                          num_timesteps,
                          device,
                          test = False,
                          num_test_runs = 1):

    env = gym.make('InvertedPendulum-v2')

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            use_gae             = True,
            use_icm             = False,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def inverted_double_pendulum_ppo(state_path,
                                 load_state,
                                 render,
                                 num_timesteps,
                                 device,
                                 test = False,
                                 num_test_runs = 1):

    #
    # TODO: this environment is a bit unstable. Let's find
    # some better settings.
    #
    env = gym.make('InvertedDoublePendulum-v2')

    #
    # Ant observations are organized as follows:
    #    Positions: 1
    #    Angles: 4
    #    Velocities: 3
    #    Contact forces: 3
    #
    actor_kw_args = {}
    actor_kw_args["activation"]   = nn.Tanh()
    actor_kw_args["split_start"]  = env.observation_space.shape[0] - 3
    actor_kw_args["hidden_left"]  = 64
    actor_kw_args["hidden_right"] = 16

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_left"]  = 128
    critic_kw_args["hidden_right"] = 128

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 1.,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = 2056,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def ant_ppo(state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test = False,
            num_test_runs = 1):

    env = gym.make('Ant-v3')

    #
    # Ant observations are organized as follows:
    #    Positions: 13
    #    Velocities: 14
    #    Contact forces: 84
    #
    actor_kw_args = {}
    actor_kw_args["activation"]   = nn.LeakyReLU()
    actor_kw_args["split_start"]  = env.observation_space.shape[0] - 84
    actor_kw_args["hidden_left"]  = 32
    actor_kw_args["hidden_right"] = 84

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_left"]  = 256
    critic_kw_args["hidden_right"] = 256

    lr     = 0.0003
    min_lr = 0.0000

    lr_dec = LinearDecrementer(
        max_iteration = 3000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 64,
            ts_per_rollout      = 2056,
            use_gae             = True,
            save_best_only      = False,
            target_kl           = 1.0,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-30., 30.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def humanoid_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False,
                 num_test_runs = 1):

    env = gym.make('Humanoid-v3')

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
    # Technically, I think actuator forces would fall under
    # proprioceptive information, but the model seems to train
    # a bit more quickly when it's coupled with the
    # exteroceptive contact forces.
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

    actor_kw_args["activation"]   = nn.Tanh()
    actor_kw_args["split_start"]  = env.observation_space.shape[0] - (84 + 23)
    actor_kw_args["hidden_left"]  = 256
    actor_kw_args["hidden_right"] = 64

    #
    # The action range for Humanoid is [-.4, .4]. Enforcing
    # this range in our predicted actions isn't required for
    # learning a good policy, but it does help speed things up.
    #
    actor_kw_args["distribution_min"] = -0.4
    actor_kw_args["distribution_max"] = 0.4

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_left"]  = 256
    critic_kw_args["hidden_right"] = 256

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = 2048,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = None,
            reward_clip         = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def humanoid_stand_up_ppo(state_path,
                          load_state,
                          render,
                          num_timesteps,
                          device,
                          test = False,
                          num_test_runs = 1):

    #
    # NOTE: this is an UNSOVLED environment.
    #
    env = gym.make('HumanoidStandup-v2')

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

    actor_kw_args["activation"]   = nn.Tanh()
    actor_kw_args["split_start"]  = env.observation_space.shape[0] - 84
    actor_kw_args["hidden_left"]  = 512
    actor_kw_args["hidden_right"] = 32

    #
    # The action range for Humanoid is [-.4, .4]. Enforcing
    # this range in our predicted actions isn't required for
    # learning a good policy, but it does help speed things up.
    #
    actor_kw_args["distribution_min"] = -0.4
    actor_kw_args["distribution_max"] = 0.4

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_left"]  = 512
    critic_kw_args["hidden_right"] = 128

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = 2048,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = None,
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def walker2d_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False,
                 num_test_runs = 1):

    env = gym.make('Walker2d-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.Tanh()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0

    max_iter = 2000000. / 2048.

    lr_dec = LinearDecrementer(
        max_iteration = max_iter,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = 2048,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def hopper_ppo(state_path,
               load_state,
               render,
               num_timesteps,
               device,
               test = False,
               num_test_runs = 1):

    env = gym.make('Hopper-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.Tanh()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0003
    min_lr = 0.0003

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 16,
            ts_per_rollout      = 2048,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            entropy_weight      = 0.0,
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def half_cheetah_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False,
                     num_test_runs = 1):

    env = gym.make('HalfCheetah-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 128

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 256

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = 2048,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)


def swimmer_ppo(state_path,
                load_state,
                render,
                num_timesteps,
                device,
                test = False,
                num_test_runs = 1):

    env = gym.make('Swimmer-v3')

    actor_kw_args = {}
    actor_kw_args["activation"]  = nn.LeakyReLU()
    actor_kw_args["hidden_size"] = 64

    critic_kw_args = actor_kw_args.copy()
    critic_kw_args["hidden_size"] = 128

    lr     = 0.0001
    min_lr = 0.0001

    lr_dec = LinearDecrementer(
        max_iteration = 1.0,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            actor_kw_args       = actor_kw_args,
            critic_kw_args      = critic_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 32,
            ts_per_rollout      = 2048,
            use_gae             = True,
            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            lr_dec              = lr_dec,
            lr                  = lr,
            min_lr              = min_lr,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            test                = test,
            num_test_runs       = num_test_runs)
