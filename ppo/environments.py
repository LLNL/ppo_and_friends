from ppo import PPO
import gym
from testing import test_policy
import numpy as np
import math
import torch
import torchvision.transforms as t_transforms
from networks import SimpleFeedForward, AtariROMNetwork

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

def run_ppo(env,
            network,
            device,
            action_type,
            lr                = 3e-4,
            lr_dec            = 1e-4,
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

def cartpole_pixels_ppo(lr,
                        use_gae,
                        use_icm,
                        state_path,
                        load_state,
                        render,
                        num_timesteps,
                        device,
                        test = False):

    env = CartPoleEnvManager()

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def cartpole_ppo(lr,
                 use_gae,
                 use_icm,
                 state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('CartPole-v0')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def pendulum_ppo(lr,
                 use_gae,
                 use_icm,
                 state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('Pendulum-v1')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "continuous",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def lunar_lander_ppo(lr,
                     use_gae,
                     use_icm,
                     state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

    env = gym.make('LunarLander-v2')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def mountain_car_ppo(lr,
                     use_gae,
                     use_icm,
                     state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

    env = gym.make('MountainCar-v0')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def mountain_car_continuous_ppo(lr,
                                use_gae,
                                use_icm,
                                state_path,
                                load_state,
                                render,
                                num_timesteps,
                                device,
                                test = False):

    env = gym.make('MountainCarContinuous-v0')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "continuous",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def acrobot_ppo(lr,
                use_gae,
                use_icm,
                state_path,
                load_state,
                render,
                num_timesteps,
                device,
                test = False):

    env = gym.make('Acrobot-v1')

    run_ppo(env               = env,
            network           = SimpleFeedForward,
            action_type       = "discrete",
            lr                = lr,
            max_ts_per_ep     = 200,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 1.0,
            test              = test)


def assault_ppo(lr,
                use_gae,
                use_icm,
                state_path,
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
            network           = AtariROMNetwork,
            action_type       = "discrete",
            lr                = lr,
            lr_dec_freq       = 150,
            max_ts_per_ep     = 200000,
            use_gae           = use_gae,
            use_icm           = use_icm,
            state_path        = state_path,
            load_state        = load_state,
            render            = render,
            num_timesteps     = num_timesteps,
            device            = device,
            ext_reward_scale  = 1.0,
            intr_reward_scale = 0.001,
            test              = test)

