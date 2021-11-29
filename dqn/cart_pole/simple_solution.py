import gym
import numpy as np
from itertools import count
from utils import get_moving_average


if __name__ == "__main__":

    num_episodes      = 1500
    top_score         = 199
    top_average       = 0.0
    top_avg_episode   = 0
    env               = gym.make("CartPole-v0")
    episode_durations = []
    bail              = False
    
    for episode_idx in range(0, num_episodes):

        state = env.reset()
    
        for ts in count():
            if state[2] + state[3] < 0:
                action = 0
            else:
                action = 1

            next_state, reward, done, _ = env.step(action)
            state = next_state
    
            if done:
                episode_durations.append(ts)
                avg = float(get_moving_average(100, episode_durations)[-1])
                print("\nEpisode: {}".format(episode_idx))
                print("    Moving average: {}".format(avg))

                top_reached = avg >= top_score
                avg_diff    = avg - top_average

                if top_reached:
                    print("Top average reached!!")
                    top_avg_episode = episode_idx
                    bail = True
                break
        if bail:
            break

    print("---------------------------------------------")
    msg  = "Top moving average: {}, reached at ".format(top_average)
    msg += "episode {}.".format(top_avg_episode)
    print(msg)
    top_count = episode_durations.count(top_score)
    print("Top score reached {} times.".format(top_count))
    
    env.close()
