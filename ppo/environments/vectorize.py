"""
    Utilities for "vectorizing" environments. As far as I can tell, this
    definition of vectorize is pretty specific to RL and refers to a way
    of reducing our environment episodes to "fixed-length trajecorty
    segments". This allows us to set our maximum timesteps per episode
    fairly low while still observing an episode from start to finish.
    This is accomplished by these "segments" that can (and often do) start
    in the middle of an episode.
    Vectorized environments are also often referred to as environment
    wrappers that contain multiple instances of environments, each of
    which can be used for training simulatneously.
"""
import numpy as np

class VectorizedEnv(object):

    def __init__(self,
                 env,
                 **kw_args):

        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space.shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        if done:
            info["terminal obsveration"] = obs
            obs = self.env.reset()

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = obs.reshape(self.observation_space.shape)
        return obs

    def render(self):
        self.env.render()
