import numpy as np
import time
import gym

num_episodes          = 1
max_steps_per_episode = 100

env = gym.make("FrozenLake-v1")

q_table = np.load("q_table.npy")

for ep_idx in range(num_episodes):
    state = env.reset()
    done  = False

    time.sleep(2)
    print("*****Epsiode {}*****".format(ep_idx))
    time.sleep(.3)
    print()
    env.render()

    for step_idx in range(max_steps_per_episode): 

        action = np.argmax(q_table[state])
        new_state, reward, done, _ = env.step(action)

        time.sleep(.3)
        print()
        env.render()

        if done:
            if reward == 1:
                print("Reached goal in {} steps.".format(step_idx))
            else:
                print("Failed to reach goal...")
            break

        state = new_state

env.close()
