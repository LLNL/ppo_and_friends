from ppo import PPO
import gym
from testing import test_policy
import numpy as np
import math
import torch
import torchvision.transforms as t_transforms
from networks import SimpleFeedForward, AtariRAMNetwork, AtariPixelNetwork

class CustomObservationSpace(object):

    def __init__(self,
                 shape):
        self.shape = shape


class CartPoleEnvManager(object):

    def __init__(self):

        self.env            = gym.make("CartPole-v0").unwrapped
        self.current_screen = None
        self.done           = False
        self.action_space   = self.env.action_space

        self.env.reset()
        screen_size = self.get_screen_height() * self.get_screen_width() * 3
        self.observation_space = CustomObservationSpace((screen_size,))

    def reset(self):
        self.env.reset()
        self.current_screen = None
        return self.get_screen_state()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def step(self, action):
        _, reward, self.done, info = self.env.step(action.item())
        obs = self.get_screen_state()
        return obs, reward, self.done, info

    def just_starting(self):
        return self.current_screen is None

    def get_screen_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = np.zeros_like(self.current_screen)
            return black_screen.flatten()
        else:
            screen_1 = self.current_screen
            screen_2 = self.get_processed_screen()
            self.current_screen = screen_2
            return (screen_2 - screen_1).flatten()

    def get_screen_height(self):
        return self.get_processed_screen().shape[2]

    def get_screen_width(self):
        return self.get_processed_screen().shape[3]

    def get_processed_screen(self):
        screen = self.render("rgb_array").transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):

        screen_height = screen.shape[1]
        top           = int(screen_height * 0.4)
        bottom        = int(screen_height * 0.8)

        screen_width  = screen.shape[2]
        left          = int(screen_width * 0.1)
        right         = int(screen_width * 0.9)
        screen        = screen[:, top : bottom, left : right]

        return screen

    def transform_screen_data(self, screen):

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)

        resize = t_transforms.Compose([
            t_transforms.ToPILImage(),
            t_transforms.Resize((40, 90)),
            t_transforms.ToTensor()])

        return resize(screen).unsqueeze(0).numpy()


#FIXME: change to inherit from gym.Wrapper
#TODO: create a pixel wrapper that stacks frames rather than uses diff.
class AtariEnvWrapper(object):

    def __init__(self,
                 env,
                 min_lives = -1):

        super(AtariEnvWrapper, self).__init__()

        self.min_lives          = min_lives
        self.env                = env
        self.action_space       = env.action_space
        self.single_state_shape = env.observation_space.shape

    def _end_game(self, done):
        return done or self.env.ale.lives() < self.min_lives


class AtariGrayScale(AtariEnvWrapper):

    def __init__(self,
                 env,
                 min_lives = -1):

        super(AtariGrayScale, self).__init__(
            env,
            min_lives)

    def rgb_to_gray_fp(self, rgb_frame):
        rgb_frame  = rgb_frame.astype(np.float32) / 255.
        gray_dot   = np.array([0.2989, 0.587 , 0.114 ], dtype=np.float32)
        gray_frame = np.expand_dims(np.dot(rgb_frame, gray_dot), axis=0)
        return gray_frame


class PixelDifferenceEnvWrapper(AtariGrayScale):

    def __init__(self,
                 env,
                 min_lives = -1):

        super(PixelDifferenceEnvWrapper, self).__init__(
            env,
            min_lives)

        self.prev_frame   = None
        self.action_space = env.action_space

        prev_shape = env.observation_space.shape
        new_shape  = (prev_shape[0], prev_shape[1], 1)
        self.observation_space = CustomObservationSpace(new_shape)

        self.reset()

    def reset(self):
        cur_frame = self.env.reset()
        cur_frame = self.rgb_to_gray_fp(cur_frame)
        self.prev_frame = cur_frame

        return self.prev_frame.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

        cur_frame = self.rgb_to_gray_fp(cur_frame)

        diff_frame      = cur_frame - self.prev_frame
        self.prev_frame = cur_frame.copy()

        done = self._end_game(done)

        return diff_frame, reward, done, info

    def render(self):
        self.env.render()


class PixelHistEnvWrapper(AtariGrayScale):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1):

        super(PixelHistEnvWrapper, self).__init__(
            env,
            min_lives)

        self.frame_cache  = None
        self.action_space = env.action_space
        self.hist_size    = hist_size

        prev_shape = env.observation_space.shape
        #TODO: since we've changed the shape, we should update this
        # to be accurate and update the networks.
        new_shape  = (prev_shape[0], prev_shape[1], hist_size)
        self.observation_space = CustomObservationSpace(new_shape)

        self.reset()

    def reset(self):
        cur_frame = self.env.reset()
        #FIXME: the ball doesn't always launch... it seems like
        # we need a left or right movement to trigger a launch.
        #cur_frame, _, _, _ = self.env.step(1)
        cur_frame = self.rgb_to_gray_fp(cur_frame)
        self.frame_cache = np.tile(cur_frame, (self.hist_size, 1, 1))

        return self.frame_cache.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

        cur_frame = self.rgb_to_gray_fp(cur_frame)

        self.frame_cache = np.roll(self.frame_cache, 1, axis=0)
        self.frame_cache[-1] = cur_frame.copy()

        done = self._end_game(done)

        return self.frame_cache, reward, done, info

    def render(self):
        self.env.render()


class RAMHistEnvWrapper(AtariEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1):

        super(RAMHistEnvWrapper, self).__init__(
            env,
            min_lives)

        ram_shape   = env.observation_space.shape
        cache_shape = (ram_shape[0] * hist_size,)

        self.observation_space = CustomObservationSpace(
            cache_shape)

        self.min_lives          = min_lives
        self.ram_size           = ram_shape[0]
        self.cache_size         = cache_shape[0]
        self.hist_size          = hist_size
        self.env                = env
        self.ram_cache          = None
        self.action_space       = env.action_space
        self.single_state_shape = env.observation_space.shape

        self.reset()

    def _reset_ram_cache(self,
                         cur_ram):
        self.ram_cache = np.tile(cur_ram, self.hist_size)

    def reset(self):
        cur_ram  = self.env.reset()
        cur_ram  = cur_ram.astype(np.float32) / 255.
        self._reset_ram_cache(cur_ram)

        return self.ram_cache.copy()

    def step(self, action):
        cur_ram, reward, done, info = self.env.step(action)
        cur_ram  = cur_ram.astype(np.float32) / 255.

        self.ram_cache = np.roll(self.ram_cache, -self.ram_size)

        offset = self.cache_size - self.ram_size
        self.ram_cache[offset :] = cur_ram.copy()

        done = self._end_game(done)

        return self.ram_cache.copy(), reward, done, info

    def render(self):
        self.env.render()


def run_ppo(env,
            network,
            device,
            action_type,
            batch_size        = 256,
            lr                = 3e-4,
            lr_dec            = 0.99,
            lr_dec_freq       = 500,
            max_ts_per_ep     = 200,
            use_gae           = False,
            use_icm           = False,
            icm_beta          = 0.8,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            render            = False,
            load_state        = False,
            state_path        = "./",
            num_timesteps     = 1,
            test              = False):

    ppo = PPO(env               = env,
              network           = network,
              device            = device,
              batch_size        = batch_size,
              action_type       = action_type,
              lr                = lr,
              lr_dec            = lr_dec,
              lr_dec_freq       = lr_dec_freq,
              max_ts_per_ep     = max_ts_per_ep,
              use_gae           = use_gae,
              use_icm           = use_icm,
              ext_reward_scale  = ext_reward_scale,
              intr_reward_scale = intr_reward_scale,
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

    wrapped_env = PixelHistEnvWrapper(
        env       = env,
        hist_size = 2,
        min_lives = 5)

    run_ppo(env               = wrapped_env,
            network           = AtariPixelNetwork,
            action_type       = "discrete",
            lr                = 0.0001,
            lr_dec_freq       = 50,
            lr_dec            = 0.9999,
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

    wrapped_env = RAMHistEnvWrapper(
        env       = env,
        hist_size = 3,
        min_lives = 5)

    run_ppo(env               = wrapped_env,
            network           = AtariRAMNetwork,
            batch_size        = 256,
            action_type       = "discrete",
            lr                = 0.0001,
            lr_dec_freq       = 20,
            lr_dec            = 0.999,
            max_ts_per_ep     = 1000,
            use_gae           = False,
            use_icm           = False,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 0.1,
            test              = test)
