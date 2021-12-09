from ppo import PPO
import gym
from testing import test_policy
import numpy as np
import math

#FIXME: let's try to remove torch
import torch
import torchvision.transforms as t_transforms

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
            action_type,
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test):

    ppo = PPO(env         = env,
              device      = device,
              action_type = action_type,
              use_gae     = use_gae,
              use_icm     = use_icm,
              render      = render,
              load_state  = load_state,
              state_path  = state_path)

    if test:
        test_policy(ppo.actor, env, render, device, action_type)
    else: 
        ppo.learn(num_timesteps)

def cartpole_pixels_ppo(use_gae,
                        use_icm,
                        state_path,
                        load_state,
                        render,
                        num_timesteps,
                        device,
                        test = False):

    env = CartPoleEnvManager()

    run_ppo(env,
            "discrete",
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)


def cartpole_ppo(use_gae,
                 use_icm,
                 state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('CartPole-v0')

    run_ppo(env,
            "discrete",
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)


def pendulum_ppo(use_gae,
                 use_icm,
                 state_path,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('Pendulum-v1')

    run_ppo(env,
            "continuous",
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)


def lunar_lander_ppo(use_gae,
                     use_icm,
                     state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

    env = gym.make('LunarLander-v2')

    run_ppo(env,
            "discrete",
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)


def mountain_car_ppo(use_gae,
                     use_icm,
                     state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test = False):

    env = gym.make('MountainCar-v0')

    run_ppo(env,
            "discrete",
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)


def mountain_car_continuous_ppo(use_gae,
                                use_icm,
                                state_path,
                                load_state,
                                render,
                                num_timesteps,
                                device,
                                test = False):

    env = gym.make('MountainCarContinuous-v0')

    run_ppo(env,
            "continuous",
            use_gae,
            use_icm,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)
