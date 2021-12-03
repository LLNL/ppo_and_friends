from ppo import PPO
import gym
import torch
import argparse
from environments import *
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--state_path", default="")
    parser.add_argument("--load_state", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_timesteps", default=500000, type=int)
    parser.add_argument("--environment", type=str, required=True,
        choices=["CartPole", "CartPolePixels", "Pendulum", "LunarLander"])

    args          = parser.parse_args()
    test          = args.test
    env_name      = args.environment
    state_path    = os.path.join(args.state_path, "saved_states", env_name)
    load_state    = args.load_state or test
    render        = args.render
    num_timesteps = args.num_timesteps

    if torch.cuda.is_available() and not test:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if env_name == "CartPole":
        cartpole_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test)

    elif env_name == "CartPolePixels":
        cartpole_pixels_ppo(state_path,
                            load_state,
                            render,
                            num_timesteps,
                            device,
                            test)

    elif env_name == "Pendulum":
        pendulum_ppo(state_path,
                     load_state,
                     render,
                     num_timesteps,
                     device,
                     test)

    elif env_name == "LunarLander":
        lunar_lander_ppo(state_path,
                         load_state,
                         render,
                         num_timesteps,
                         device,
                         test)
