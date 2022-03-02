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
#TODO: let's support multiple environment instances for parallel training.
# We should probably have options for both HPC clusters and cpu-only.

import numpy as np
from .env_wrappers import IdentityWrapper

class VectorizedEnv(IdentityWrapper):

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize our vectorized environment.

            Arguments:
                env    The environment to vectorize.
        """
        super(VectorizedEnv, self).__init__(
            env,
            **kw_args)

    def step(self, action):
        """
            Take a step in our environment with the given action.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, done, and info
                tuple.
        """
        obs, reward, done, info = self.env.step(action)

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space.shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        if done:
            info["terminal observation"] = obs.copy()
            obs = self.env.reset()
            obs = obs.reshape(self.observation_space.shape)

        return obs, reward, done, info
