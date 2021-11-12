import numpy as np
import gym
import random
import time

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size  = env.observation_space.n

print("Action space size: {}".format(action_space_size))
print("State space size: {}".format(state_space_size))

q_table = np.zeros((state_space_size, action_space_size))

num_episodes           = 100000
max_steps_per_episode  = 100
learning_rate          = 0.1
discount_rate          = 0.99
exploration_rate       = 1.0
min_exploration_rate   = 0.01
max_exploration_rate   = 1.0
exploration_decay_rate = 0.001
all_episode_rewards    = []

#
# Iterate through our episodes.
#
for ep_idx in range(num_episodes):

    state = env.reset()
    done  = False
    episode_reward = 0

    for step in range(max_steps_per_episode):

        #
        # Determine if we're going to be greedy (take the optimal route) or
        # explore.
        #
        greedy_value = np.random.uniform(0.0, 1.0)

        if greedy_value > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        #
        # Perform our action and observe the results.
        #
        new_state, reward, done, info = env.step(action)

        #
        # Update our Q-table using the learning rate (weighted sum).
        #
        q_table[state, action] = (((1.0 - learning_rate) * q_table[state, action]) +
            (learning_rate * (reward + (discount_rate * np.max(q_table[new_state])))))

        state           = new_state
        episode_reward += reward

        if done:
            break

    #
    # Exponentially decay our exploration rate based one episode count.
    #
    exploration_rate = (min_exploration_rate +
        ((max_exploration_rate - min_exploration_rate) *
          np.exp(-exploration_decay_rate * ep_idx)))

    all_episode_rewards.append(episode_reward)


#
# Look at the results.
#
all_episode_rewards = np.array(all_episode_rewards)

print("Final exploration rate: {}".format(exploration_rate))
print("Max reward: {}".format(max(all_episode_rewards)))

average_window = int(num_episodes / 10)
print("Average reward per {} episodes:".format(average_window))

for i in range(0, num_episodes - average_window, average_window):
    print("    {} -> {}: {}".format(i, i + average_window,
        np.mean(all_episode_rewards[i : (i + average_window)])))

print("\nQ table: ")
print(q_table)

np.save("q_table.npy", q_table)
