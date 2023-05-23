"""
    Wrappers that are used to allow an environment built for version
    X to work in gym version Y.
"""
import gymnasium as gym
import numpy as np

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


def gym_space_to_gymnasium_space(space):
    """
        gym and gymnasium spaces are incompatible. This function
        just converts gym spaces to gymnasium spaces to bypass
        the errors that crop up.

        Arguments:
            space     The gym space to convert.

        Returns:
            The input space converted to gymnasium.
    """
    import gym as old_gym
    if type(space) == old_gym.spaces.Box:
        space = gym.spaces.Box(
            low   = space.low,
            high  = space.high,
            shape = space.shape,
            dtype = space.dtype)

    elif type(space) == old_gym.spaces.Discrete:
        space = gym.spaces.Discrete(
            n     = space.n,
            start = space.start)

    elif type(space) == old_gym.spaces.MultiBinary:
        space = gym.spaces.MultiBinary(
            n     = space.n)

    elif type(space) == old_gym.spaces.MultiDiscrete:
        space = gym.spaces.MultiDiscrete(
            nvec  = space.nvec,
            dtype = space.dtype)

    elif type(space) == old_gym.spaces.Dict:
        new_space = gym.spaces.Dict()

        for key in space:
            new_space[key] = gym_space_to_gymnasium_space(space[key])

        space = new_space

    elif type(space) == old_gym.spaces.Tuple:
        new_space = []

        for subspace in space:
            new_space.append(gym_space_to_gymnasium_space(subspace))

        space = gym.spaces.Tuple(new_space)

    else:
        msg  = "WARNING: skipping conversion of space "
        msg += f"{type(self.action_space[agent_id])}."
        print(msg)

    return space

