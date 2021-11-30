from ppo import PPO
import gym
import torch

env = gym.make('Pendulum-v1')
model = PPO(env, torch.device("cuda"))
model.learn(3)
