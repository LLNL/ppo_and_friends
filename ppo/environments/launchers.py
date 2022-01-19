from ppo import PPO
import gym
from testing import test_policy
from networks import SimpleFeedForward, AtariRAMNetwork, AtariPixelNetwork
from networks import SimpleSplitObsNetwork
from networks import ICM, LinearObservationEncoder, Conv2dObservationEncoder_orig
from .gym_wrappers import *
import torch.nn as nn
from utils.decrementers import *


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
            ac_kw_args          = {},
            icm_kw_args         = {},
            surr_clip           = 0.2,
            bootstrap_clip      = (-10.0, 10.0),
            dynamic_bs_clip     = False,
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
              ac_kw_args         = ac_kw_args,
              surr_clip          = surr_clip,
              bootstrap_clip     = bootstrap_clip,
              dynamic_bs_clip    = dynamic_bs_clip,
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


def cartpole_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False,
                 num_test_runs = 1):

    env = gym.make('CartPole-v0')

    lr     = 0.0003
    min_lr = 0.000090

    lr_dec = LinearDecrementer(
        max_iteration  = 1000,
        max_value      = lr,
        min_value      = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 200,
            use_gae            = True,
            use_icm            = False,
            normalize_obs      = True,
            normalize_rewards  = True,
            obs_clip           = (-10., 10.),
            reward_clip        = (-10., 10.),
            dynamic_bs_clip    = False,
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

    lr     = 0.0003
    min_lr = 0.000090

    lr_dec = LinearDecrementer(
        max_iteration = 1000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
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
    ac_kw_args = {"activation" : nn.LeakyReLU()}

    lr     = 0.0003
    min_lr = 0.000090

    lr_dec = LinearDecrementer(
        max_iteration = 200,
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
            ac_kw_args          = ac_kw_args,
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
    # Lunar lnader observations are organized as follows:
    #    Positions: 2
    #    Positional velocities: 2
    #    Angle: 1
    #    Angular velocities: 1
    #    Leg contact: 2
    #
    ac_kw_args = {}
    #ac_kw_args["split_start"]  = env.observation_space.shape[0] - 2
    #ac_kw_args["hidden_left"]  = 32
    #ac_kw_args["hidden_right"] = 8

    #
    # Extra args for the actor critic models.
    # I find that leaky relu does much better with the lunar
    # lander env.
    #
    ac_kw_args["activation"]  = nn.LeakyReLU()
    ac_kw_args["hidden_size"] = 64

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 4000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleFeedForward,
            max_ts_per_ep       = 32,
            ts_per_rollout      = 2048,
            batch_size          = 512,
            ac_kw_args          = ac_kw_args,
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


def mountain_car_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False,
                     num_test_runs = 1):

    env = gym.make('MountainCar-v0')

    lr     = 0.0003
    min_lr = 0.0002

    lr_dec = LogDecrementer(
        max_iteration = 8000,
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
            use_icm            = True,
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
    # Leaky relu tends to work well here.
    #
    ac_kw_args = {"activation" : nn.LeakyReLU()}

    lr     = 0.0003
    min_lr = 0.00014

    lr_dec = LogDecrementer(
        max_iteration = 8000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 32,
            batch_size         = 256,
            ts_per_rollout     = 2048,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            ac_kw_args         = ac_kw_args,
            use_gae            = True,
            use_icm            = True,
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


def acrobot_ppo(state_path,
                load_state,
                render,
                num_timesteps,
                device,
                test = False,
                num_test_runs = 1):

    env = gym.make('Acrobot-v1')

    lr     = 0.0003
    min_lr = 0.00015

    lr_dec = LogDecrementer(
        max_iteration = 8000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                = env,
            ac_network         = SimpleFeedForward,
            max_ts_per_ep      = 100,
            ts_per_rollout     = 2048,
            lr_dec             = lr_dec,
            lr                 = lr,
            min_lr             = min_lr,
            use_gae            = True,
            use_icm            = True,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


def assault_ram_ppo(state_path,
                    load_state,
                    render,
                    num_timesteps,
                    device,
                    test = False,
                    num_test_runs = 1):

    if test and render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env = gym.make(
            'Assault-ram-v0',
            render_mode='human')
    else:
        env = gym.make(
            'Assault-ram-v0')

    run_ppo(env                = env,
            ac_network         = AtariRAMNetwork,
            lr                 = 0.0001,
            max_ts_per_ep      = 1000,
            use_gae            = True,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


#FIXME: obs space is broken
def assault_pixels_ppo(state_path,
                       load_state,
                       render,
                       num_timesteps,
                       device,
                       test = False,
                       num_test_runs = 1):

    if test and render:
        #
        # NOTE: we don't want to explicitly call render for atari games.
        # They have more advanced ways of rendering.
        #
        render = False

        env = gym.make(
            'Assault-v0',
            render_mode='human')
    else:
        env = gym.make(
            'Assault-v0')

    wrapped_env = PixelHistEnvWrapper(
        env       = env,
        hist_size = 2,
        min_lives = 5)

    run_ppo(env                = wrapped_env,
            ac_network         = AtariPixelNetwork,
            lr                 = 0.0001,
            max_ts_per_ep      = 10000,
            use_gae            = True,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


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

    lr     = 0.0003
    min_lr = 0.0

    lr_dec = LinearDecrementer(
        max_iteration = 7000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                  = wrapped_env,
            ac_network           = AtariPixelNetwork,
            batch_size           = 512,
            ts_per_rollout       = 2048,
            max_ts_per_ep        = 64,
            epochs_per_iter      = 30,
            reward_clip         = (-1., 1.),
            bootstrap_clip      = (-1., 1.),
            target_kl           = 0.015,
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
        env       = env,
        hist_size = 3,
        min_lives = 5)

    run_ppo(env                = wrapped_env,
            ac_network         = AtariRAMNetwork,
            batch_size         = 512,
            lr                 = 0.0001,
            max_ts_per_ep      = 20000,
            use_gae            = True,
            save_best_only     = True,
            state_path         = state_path,
            load_state         = load_state,
            render             = render,
            num_timesteps      = num_timesteps,
            device             = device,
            test               = test,
            num_test_runs      = num_test_runs)


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
    ac_kw_args = {}
    ac_kw_args["split_start"]  = env.observation_space.shape[0] - 10
    ac_kw_args["hidden_left"]  = 64
    ac_kw_args["hidden_right"] = 64

    lr     = 0.0003
    min_lr = 0.0000009

    lr_dec = LogDecrementer(
        max_iteration = 12000,
        max_value     = lr,
        min_value     = min_lr)

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            ac_kw_args          = ac_kw_args,
            batch_size          = 512,
            max_ts_per_ep       = 64,
            ts_per_rollout      = 2048,
            use_gae             = True,
            use_icm             = False,
            save_best_only      = True,
            epochs_per_iter     = 20,
            mean_window_size    = 200,
            target_kl           = 0.015,

            normalize_obs       = True,
            normalize_rewards   = True,
            obs_clip            = (-10., 10.),
            reward_clip         = (-10., 10.),
            bootstrap_clip      = (-10., 10.),

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
    ac_kw_args = {}
    ac_kw_args["split_start"]  = env.observation_space.shape[0] - 84
    ac_kw_args["hidden_left"]  = 64
    ac_kw_args["hidden_right"] = 128

    run_ppo(env                 = env,
            ac_network          = SimpleSplitObsNetwork,
            ac_kw_args          = ac_kw_args,
            batch_size          = 256,
            max_ts_per_ep       = 64,
            ts_per_rollout      = 1024,
            use_gae             = True,
            use_icm             = False,
            save_best_only      = False,
            epochs_per_iter     = 20,
            mean_window_size    = 500,
            target_kl           = 0.05,
            lr                  = 0.0003,
            min_lr              = 0.000095,
            #lr_dec              = 0.999,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            intr_reward_weight  = 10.0,
            entropy_weight      = 0.01,
            test                = test,
            num_test_runs       = num_test_runs)
