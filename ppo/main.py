from ppo import PPO
import gym
import torch
import argparse
from test import test_policy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",
        choices=["train", "test"])
    parser.add_argument("--model_path", default="./saved_models")
    parser.add_argument("--load_state", action="store_true")
    parser.add_argument("--render_test", action="store_true")

    args        = parser.parse_args()
    mode        = args.mode
    model_path  = args.model_path
    load_state  = args.load_state or mode == "test"
    render_test = args.render_test

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env = gym.make('Pendulum-v1')

    ppo = PPO(env          = env,
              device       = device,
              load_weights = load_state,
              model_path   = model_path)

    if mode == "train":
        ppo.learn(100000)

    elif mode == "test":
        test_policy(ppo.actor, env, render_test, device)
