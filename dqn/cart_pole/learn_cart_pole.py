import gym
import json
import os
import numpy as np
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
from utils import Experience
from networks import LinearDQN
from networks import ConvDQN
from agents import DQNAgent
from utils import CartPoleEnvManager
from utils import plot
from utils import EpsilonGreedyStrategy
from utils import ReplayMemory
from utils import extract_tensors
from utils import QValues
import torch.nn.functional as t_functional


save_path     = "saved_models"
batch_size    = 256
gamma         = 0.999
eps_start     = 1.
eps_stop      = 0.01
eps_decay     = 0.001
target_update = 10
memory_size   = 100000
lr            = 0.001
num_episodes  = 300

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

env_manager = CartPoleEnvManager(device)
strategy    = EpsilonGreedyStrategy(eps_start, eps_stop, eps_decay)
agent       = DQNAgent(strategy, env_manager.num_actions_available(), device)
memory      = ReplayMemory(memory_size)
episode_durations = []

policy_net = LinearDQN(
    env_manager.get_screen_height(), 
    env_manager.get_screen_width()).to(device)

target_net = LinearDQN(
    env_manager.get_screen_height(), 
    env_manager.get_screen_width()).to(device)

#policy_net = ConvDQN(
#    env_manager.get_screen_height(), 
#    env_manager.get_screen_width()).to(device)
#
#target_net = ConvDQN(
#    env_manager.get_screen_height(), 
#    env_manager.get_screen_width()).to(device)

state_dict = {"current_episode" : 0}
state_file = os.path.join(save_path, "state_dict.json")

if os.path.exists(os.path.join(save_path, target_net.name + ".model")):
    print("Loading weights from saved model.")

    policy_net.load(save_path)

    if os.path.exists(state_file):
        with open(state_file, "r") as in_f:
            state_dict = json.load(in_f)
else:
    os.makedirs(save_path, exist_ok=True)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params = policy_net.parameters(), lr = lr)

starting_episode   = state_dict["current_episode"]
agent.current_step = starting_episode

print("Starting from episode: {}".format(starting_episode))

for episode_idx in range(starting_episode, starting_episode + num_episodes):
    env_manager.reset()
    state = env_manager.get_state()

    for ts in count():
        action     = agent.select_action(state, policy_net)
        reward     = env_manager.take_action(action)
        next_state = env_manager.get_state()

        memory.push(Experience(state, action, next_state, reward))

        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)

            states, actions, rewards, next_states = \
                extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_actions     = torch.argmax(policy_net(next_states), dim=1)

            next_q_values    = QValues.get_next(target_net,
                next_states, next_actions)
            target_q_values  = (next_q_values * gamma) + rewards

            loss = t_functional.mse_loss(current_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if env_manager.done:
            episode_durations.append(ts)
            plot(episode_durations, 100, episode_idx,
                strategy.get_exploration_rate(agent.current_step))
            break

    target_net.save(save_path)

    state_dict["current_episode"] = episode_idx

    with open(state_file, "w") as out_f:
        json.dump(state_dict, out_f)

    if episode_idx % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

env_manager.close()
