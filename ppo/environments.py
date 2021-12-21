from ppo import PPO
import gym
from testing import test_policy
from networks import SimpleFeedForward, AtariRAMNetwork, AtariPixelNetwork
from networks import ICM, LinearObservationEncoder, Conv2dObservationEncoder_orig
from custom_environments.gym_wrappers import *


def run_ppo(env,
            network,
            device,
            action_type,
            icm_network         = ICM,
            icm_encoder         = LinearObservationEncoder,
            batch_size          = 256,
            timesteps_per_batch = 2048,
            epochs_per_iter     = 10,
            lr                  = 3e-4,
            min_lr              = 1e-4,
            lr_dec              = 0.99,
            lr_dec_freq         = 500,
            max_ts_per_ep       = 200,
            use_gae             = False,
            use_icm             = False,
            icm_beta            = 0.8,
            ext_reward_scale    = 1.0,
            intr_reward_scale   = 1.0,
            entropy_weight      = 0.01,
            render              = False,
            load_state          = False,
            state_path          = "./",
            num_timesteps       = 1,
            test                = False):

    ppo = PPO(env               = env,
              network           = network,
              icm_network       = icm_network,
              icm_encoder       = icm_encoder,
              device            = device,
              batch_size        = batch_size,
              action_type       = action_type,
              lr                = lr,
              min_lr            = min_lr,
              lr_dec            = lr_dec,
              lr_dec_freq       = lr_dec_freq,
              max_ts_per_ep     = max_ts_per_ep,
              use_gae           = use_gae,
              use_icm           = use_icm,
              ext_reward_scale  = ext_reward_scale,
              intr_reward_scale = intr_reward_scale,
              entropy_weight    = entropy_weight,
              render            = render,
              load_state        = load_state,
              state_path        = state_path)

    if test:
        test_policy(ppo.actor, env, render, device, action_type)
    else: 
        ppo.learn(num_timesteps)

def cartpole_pixels_ppo(state_path,
                        load_state,
                        render,
                        num_timesteps,
                        device,
                        test = False):

    env = CartPoleEnvManager()

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            max_ts_per_ep     = 200,
            use_gae           = True,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def cartpole_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('CartPole-v0')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            max_ts_per_ep     = 200,
            use_gae           = True,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def pendulum_ppo(state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('Pendulum-v1')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "continuous",
            max_ts_per_ep     = 200,
            use_gae           = False,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def lunar_lander_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

    env = gym.make('LunarLander-v2')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            max_ts_per_ep     = 1001,
            use_gae           = True,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def mountain_car_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

    env = gym.make('MountainCar-v0')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            max_ts_per_ep     = 200,
            use_gae           = True,
            use_icm           = True,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            entropy_weight    = 0.01,
            test              = test)


def mountain_car_continuous_ppo(state_path,
                                load_state,
                                render,
                                num_timesteps,
                                device,
                                test = False):

    env = gym.make('MountainCarContinuous-v0')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "continuous",
            max_ts_per_ep     = 200,
            use_gae           = True,
            use_icm           = True,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            entropy_weight    = 0.01,
            test              = test)


def acrobot_ppo(state_path,
                load_state,
                render,
                num_timesteps,
                device,
                test = False):

    env = gym.make('Acrobot-v1')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            max_ts_per_ep     = 200,
            use_gae           = True,
            use_icm           = True,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def assault_ram_ppo(state_path,
                    load_state,
                    render,
                    num_timesteps,
                    device,
                    test = False):

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

    run_ppo(env               = env,
            network           = AtariRAMNetwork,
            action_type       = "discrete",
            lr                = 0.0001,
            lr_dec_freq       = 30,
            lr_dec            = 0.95,
            #max_ts_per_ep     = 200000,
            max_ts_per_ep     = 1000,
            use_gae           = True,
            use_icm           = True,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 0.01,
            test              = test)


def assault_pixels_ppo(state_path,
                       load_state,
                       render,
                       num_timesteps,
                       device,
                       test = False):

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

    run_ppo(env               = wrapped_env,
            network           = AtariPixelNetwork,
            action_type       = "discrete",
            lr                = 0.0001,
            lr_dec_freq       = 30,
            lr_dec            = 0.99,
            max_ts_per_ep     = 10000,
            use_gae           = True,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 0.01,
            test              = test)


def breakout_pixels_ppo(state_path,
                        load_state,
                        render,
                        num_timesteps,
                        device,
                        test = False):

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
        env       = env,
        hist_size = 3,
        min_lives = 5)

    run_ppo(env                 = wrapped_env,
            network             = AtariPixelNetwork,
            batch_size          = 64,
            timesteps_per_batch = 2048,
            epochs_per_iter     = 10,
            action_type         = "discrete",
            lr                  = 0.0002,
            min_lr              = 0.000095,
            lr_dec_freq         = 2,
            lr_dec              = 0.95,
            max_ts_per_ep       = 1000,
            use_gae             = True,
            use_icm             = False,
            state_path          = state_path,
            load_state          = load_state,
            render              = render,
            num_timesteps       = num_timesteps,
            device              = device,
            ext_reward_scale    = 1.0,
            intr_reward_scale   = 0.1,
            entropy_weight      = 0.01,
            test                = test)


def breakout_ram_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

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

    run_ppo(env               = wrapped_env,
            network           = AtariRAMNetwork,
            batch_size        = 512,
            action_type       = "discrete",
            lr                = 0.0001,
            lr_dec_freq       = 100,
            lr_dec            = 0.9999,
            max_ts_per_ep     = 20000,
            use_gae           = True,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            entropy_weight    = 0.01,
            test              = test)
