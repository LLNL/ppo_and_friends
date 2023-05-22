"""
    Wrappers that are used to allow an environment built for version
    X to work in gym version Y.
"""
import gymnasium as gym
import numpy as np

#FIXME: no longer needed?
class Gym21To26():
    """
        There are some big changes between Gym versions .21 and
        .26. Some environments can be converted with a simple wrapper.
    """
    def __init__(self, env):
        """
            Initialize the wrapper.
        """
        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.ale               = getattr(env, "ale", None)
        self.spec              = getattr(env, "spec", None)

    def seed(self, seed):
        """
            Set the seed. This changed in .26.

            Arguments:
                seed    The seed to set.
        """
        self.env.seed(seed)

    def step(self, actions, **kw_args):
        """
            Step through the environment.

            Arguments:
                actions    The actions to take.

            Returns:
                A tuple containing obs, reward, terminated, truncated, 
                and info.
        """
        obs, reward, terminated, info = self.env.step(actions)

        truncated = np.zeros(len(terminated)).astype(bool)

        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kw_args):
        """
            Reset the environment.

            Returns:
                The obs and info.
        """
        obs = self.env.reset()
        return obs, {}

    def render(self, *args, **kw_args):
        """
            Render the environment.
        """
        return self.env.render(*args, **kw_args)
