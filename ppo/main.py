import gym
import torch
import argparse
from environments import *
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--state_path", default="")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_timesteps", default=1000000, type=int)
    parser.add_argument("--environment", "-e", type=str, required=True,
        choices=["CartPole", "CartPolePixels", "Pendulum", "LunarLander",
                 "MountainCar"])

    args          = parser.parse_args()
    test          = args.test
    env_name      = args.environment
    state_path    = os.path.join(args.state_path, "saved_states", env_name)
    clobber       = args.clobber
    render        = args.render
    num_timesteps = args.num_timesteps

    load_state    = not clobber or test

    if torch.cuda.is_available() and not test:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if env_name == "CartPole":
        cartpole_ppo(False,
                     state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test)

    elif env_name == "CartPolePixels":
        cartpole_pixels_ppo(False,
                            state_path,
                            load_state,
                            render,
                            num_timesteps,
                            device,
                            test)

    elif env_name == "Pendulum":
        pendulum_ppo(False,
                     state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test)

    elif env_name == "LunarLander":
        lunar_lander_ppo(False,
                         state_path,
                         load_state,
                         render,
                         num_timesteps,
                         device,
                         test)

    elif env_name == "MountainCar":
        mountain_car_ppo(False,
                         state_path,
                         load_state,
                         render,
                         num_timesteps,
                         device,
                         test)
