"""
    A home for OpenAI Gym wrappers. These wrappers should all be
    specific to environments found in OpenAI Gym.
"""
import sys
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import cv2
from abc import ABC
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

#DEBUGGING SUPPORT
def show_frame(frame_cache):
    from PIL import Image
    first = frame_cache[-1].squeeze()
    first = (first * 255.).astype(np.uint8)
    img = Image.fromarray(first, 'L')
    img.show()

    last = frame_cache[0].squeeze()
    last = (last * 255.).astype(np.uint8)
    img = Image.fromarray(last, 'L')
    img.show()

    input("")

################################################################################
#                       General Atari Wrappers                                 #
################################################################################

class AtariEnvWrapper(ABC):
    """
    A base class generic wrapper for atari games.

    Some important concepts:
        1. While gym environments do offer frame skip, they don't
           allow access to the skipped frames, which is important.
           This class re-implements frame skipping in such a way
           that allows access to these frames.
        2. false_done_reset and true_done_reset are actions that
           are taken under the following conditions, respectively:
               a. A life is lost, but the game is not over, i.e
                  allow_life_loss is set to False, which results
                  in a "done" state, but we don't end the game.
               b. A life is lost, and the game is over.
    """

    def __init__(self,
                 env,
                 allow_life_loss = False,
                 skip_k_frames   = 1,
                 **kw_args):
        """
        Parameters:
        -----------
        env: gymnasium env
            The atari environment to wrap.
        allow_life_loss: bool
            If True, the game will only end when you've
            lost your last life. If False, the game will
            end after losing any lives.
        skip_k_frames: int
            The number of frames to skip. In reality, this
            is a misleading name. A value of 1 means no
            frames are skipped. A value of 2 means 1 frame
            is skipped. I'm only following this convention
            because it's what's used in papers...
        """
        self.allow_life_loss   = allow_life_loss
        self.life_count        = env.ale.lives()
        self.env               = env
        self.action_space      = env.action_space
        self.skip_k_frames     = skip_k_frames
        self.true_done         = True

    def false_done_reset(self):
        """
        What to do when we've acceptably lost a life, but we haven't
        ended the game.
        """
        #
        # NOTE: 0 is generally a NOOP action, but I'm not sure that's
        # guaranteed...
        #
        obs, _, _, _, info = self.env.step(0)
        return obs, info

    def reset(self, *args, **kw_args):
        """
        Reset the environment.
        """
        raise NotImplementedError

    def true_done_reset(self):
        """
        The action we should take when we are truly done, not just
        when we've acceptable lost a life. This will be called by
        the state_dependent_reset parent method.
        """
        return self.env.reset()

    def _frame_skip_step(self,
                         action,
                         step_func = None):
        """
        Take a step in the environment while skipping frames
        if requested.

        Parameters:
        -----------
        action: array-like or number
            The action to take. This will be repeated for every
            skipped frame.
        step_func: function
            An optional function to perform the actual step.
            If none, the environment's step function will be
            used. NOTE: this is an important argument, as it
            allows you to step in a way that has access to the
            skipped frames.

        Returns:
        --------
        tuple:
            A tuple of form (observation, reward, terminated,
            truncated, inf), where reward
            is actually the sum of all rewards from the frames that were
            stepped through.
        """

        if step_func == None:
            step_fun = self.env.step

        k_reward_sum = 0
        terminated   = False
        life_lost    = False

        for k in range(self.skip_k_frames):
            obs, reward, s_terminated, truncated, info = step_func(action)

            k_reward_sum += reward
            terminated    = terminated or s_terminated
            life_lost     = life_lost or info["life lost"]

            #
            # We lost a life, but we're not done. We might need to
            # take some actions here. Tricky business: we typically
            # set allow_life_loss to True when testing and false when
            # training. This allows for false "done" states when
            # training but not when testing. Calling reset() here
            # will rely on the child class to know what it should do
            # in either case.
            #
            if not terminated and self.allow_life_loss and info["life lost"]:
                obs, new_info = self.reset()
                info.update(new_info)

        info["life lost"] = life_lost

        return obs, k_reward_sum, terminated, truncated, info

    def _check_if_done(self, done):
        """
        Determine whether or not we're "done". This is a bit tricky here;
        if we've lost a life, we say that we're done, but we don't reset
        the environment. We considered this a "false done". This method
        will return whether or not we're truly done or falsely done, and
        it will set a flag that says whether or not we're truly done.

        Parameters:
        -----------
        done: bool
            Whether or not the environment is actually done and
            requires a reset.

        Returns:
        --------
        bool:
            Whether or not we're truly or falsely done. In other words,
            did we lose any lives?
        """
        self.true_done = done

        life_lost = False
        if self.env.ale.lives() < self.life_count:
            life_lost = True

        life_loss_done = False
        if not self.allow_life_loss and life_lost:
            life_loss_done = True

        self.life_count = self.env.ale.lives()

        return (done or life_loss_done, life_lost)

    def _state_dependent_reset(self):
        """
        Perform any needed actions for a reset. Again, this is tricky.
        If our environment is truly done, we need to perform a reset.
        If we're not done, we're allowing life loss, and we've lost a life,
        then we don't reset the environment, but we may need to take an
        action (like firing a ball).

        Returns:
        --------
        tuple:
            (obs, info, true_done)
        """
        true_done = self.true_done

        if self.true_done:
            self.true_done = False
            obs, info = self.true_done_reset()
        else:
            obs, info = self.false_done_reset()

        self.lives = self.env.ale.lives()
        return obs, info, true_done

    def seed(self, seed):
        """
        Set the environment's random seed.

        Parameters:
        -----------
        seed: int or None
            The random seed.
        """
        self.env.seed(seed)


class AtariPixels(AtariEnvWrapper):
    """
    A generic wrapper for atari games with pixel observations.
    """

    def __init__(self,
                 env,
                 allow_life_loss = False,
                 frame_size      = 84,
                 **kw_args):
        """
        Parameters:
        -----------
        env: gymnasium env
            The environment to wrap.
        allow_life_loss: bool
            If True, the game will only end when you've
            lost your last life. If False, the game will
            end after losing any lives.
        frame_size: int
            The pixel frame size to enforce.
        """

        super(AtariPixels, self).__init__(
            env,
            allow_life_loss,
            **kw_args)

        prev_shape      = env.observation_space.shape
        self.h_start    = 0
        self.h_stop     = prev_shape[0]
        self.w_start    = 0
        self.w_stop     = prev_shape[1]
        self.frame_size = frame_size

        new_shape = (1, frame_size, frame_size)
        low       = np.zeros(new_shape)
        high      = np.full(new_shape, 1.0)

        self.observation_space = Box(
            low   = low,
            high  = high,
            shape = new_shape,
            dtype = np.float32)

    def rgb_to_gray(self, rgb_frame):
        """
        Convert an RGB frame to grayscale.

        Parameters:
        -----------
        rgb_frame: np.ndarray
             An array containing RGB pixel information.

        Returns:
        --------
        np.ndarray:
            A grayscale version of the input frame.
        """
        rgb_frame  = rgb_frame.astype(np.float32) / 255.
        gray_dot   = np.array([0.2989, 0.587 , 0.114 ], dtype=np.float32)
        gray_frame = np.expand_dims(np.dot(rgb_frame, gray_dot), axis=0)

        return gray_frame

    def crop_frame(self,
                   frame,
                   h_start = None,
                   h_stop  = None,
                   w_start = None,
                   w_stop  = None):
        """
        Crop a given frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to crop.
        h_start: int or None
            Height start.
        h_stop: int or None
            Height stop.
        w_start: int or None
            Width start.
        w_stop: int or None
            Width stop.

        Returns:
        --------
        np.ndarray:
            The cropped frame.
        """

        h_start = self.h_start if h_start == None else h_start
        h_stop  = self.h_stop if h_stop == None else h_stop
        w_start = self.w_start if w_start == None else w_start
        w_stop  = self.w_stop if w_stop == None else w_stop

        return frame[:, h_start : h_stop, w_start : w_stop]

    def resize_frame(self,
                     frame,
                     shape  = None,
                     interp = cv2.INTER_AREA):
        """
        Resize a given frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to resize.
        shape: tuple or None
            The new shape for the frame.
        interp: cv2 interpolation method
            Which method to use for interpolation

        Returns:
        --------
        np.ndarray:
            A resized version of the input frame.
        """

        #
        # c2v is reversed.
        #
        os_shape = (self.observation_space.shape[2],
                    self.observation_space.shape[1])
        shape = os_shape if shape == None else shape

        return cv2.resize(frame, shape, interpolation = interp)

    def rgb_to_processed_frame(self, rgb_frame):
        """
        Process an RGB frame into an observation frame.

        Parameters:
        -----------
        rgb_frame: np.ndarray
            The input RGB frame.

        Returns:
        --------
        np.ndarray:
            The resulting processed frame.
        """
        new_frame = self.rgb_to_gray(rgb_frame)
        new_frame = self.crop_frame(new_frame)
        new_frame = np.expand_dims(self.resize_frame(new_frame.squeeze()), 0)
        return new_frame


class PixelHistEnvWrapper(AtariPixels):

    def __init__(self,
                 env,
                 hist_size         = 2,
                 allow_life_loss   = False,
                 use_frame_pooling = True,
                 punish_end        = False,
                 **kw_args):

        super(PixelHistEnvWrapper, self).__init__(
            env             = env,
            allow_life_loss = allow_life_loss,
            **kw_args)

        self.frame_cache       = None
        self.prev_frame        = None
        self.action_space      = env.action_space
        self.hist_size         = hist_size
        self.use_frame_pooling = use_frame_pooling
        self.punish_end        = punish_end

    def reset(self, *args, **kw_args):
        cur_frame, info, true_done = self._state_dependent_reset()
        cur_frame = self.rgb_to_processed_frame(cur_frame)

        if true_done:
            self.frame_cache = np.tile(cur_frame, (self.hist_size, 1, 1))

        self.prev_frame  = cur_frame.copy()

        return self.frame_cache.copy(), info

    def _env_step(self, action):
        cur_frame, reward, terminated, truncated, info = self.env.step(action)

        cur_frame = self.rgb_to_processed_frame(cur_frame)

        #
        # If we're using frame pooling, take the max pixel value
        # between the current and last frame. We need to keep track
        # of the prev frame separately from the cache, otherwise we
        # end up with pixel "streaks".
        #
        if self.use_frame_pooling:
            cur_copy  = cur_frame.copy()
            cur_frame = np.maximum(cur_frame, self.prev_frame)
            self.prev_frame = cur_copy

        self.frame_cache = np.roll(self.frame_cache, -1, axis=0)
        self.frame_cache[-1] = cur_frame.copy()

        terminated, life_lost   = self._check_if_done(terminated)
        info["life lost"] = life_lost

        return self.frame_cache, reward, terminated, truncated, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self._frame_skip_step(
            action    = action,
            step_func = self._env_step)

        if self.punish_end:
            #
            # Return a negative reward for failure.
            #
            if terminated and reward == 0:
                reward = -1.

        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)


class RAMHistEnvWrapper(AtariEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 allow_life_loss = False,
                 punish_end      = False,
                 **kw_args):

        super(RAMHistEnvWrapper, self).__init__(
            env             = env,
            allow_life_loss = allow_life_loss,
            **kw_args)

        ram_shape   = env.observation_space.shape
        cache_shape = (ram_shape[0] * hist_size,)

        low       = np.zeros(cache_shape)
        high      = np.full(cache_shape, 1.0)

        self.observation_space = Box(
            low   = low,
            high  = high,
            shape = cache_shape,
            dtype = np.float32)

        self.ram_size           = ram_shape[0]
        self.cache_size         = cache_shape[0]
        self.hist_size          = hist_size
        self.env                = env
        self.ram_cache          = None
        self.action_space       = env.action_space
        self.punish_end         = punish_end

    def _reset_ram_cache(self,
                         cur_ram):
        self.ram_cache = np.tile(cur_ram, self.hist_size)

    def reset(self, *args, **kw_args):
        cur_ram, info, true_done = self._state_dependent_reset()
        cur_ram = cur_ram.astype(np.float32) / 255.

        #
        # NOTE: this wrapper has 'soft-resets' baked into it.
        #
        if true_done:
            self._reset_ram_cache(cur_ram)

        return self.ram_cache.copy(), info

    def _env_step(self, action):
        cur_ram, reward, terminated, truncated, info = self.env.step(action)
        cur_ram  = cur_ram.astype(np.float32) / 255.

        self.ram_cache = np.roll(self.ram_cache, -self.ram_size)

        offset = self.cache_size - self.ram_size
        self.ram_cache[offset :] = cur_ram.copy()

        terminated, life_lost = self._check_if_done(terminated)
        info["life lost"] = life_lost

        return self.ram_cache.copy(), reward, terminated, truncated, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._frame_skip_step(
            action    = action,
            step_func = self._env_step)

        if self.punish_end:
            #
            # Return a negative reward for failure.
            #
            if terminated and reward == 0:
                reward = -1.

        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

################################################################################
#                       Atari Breakout Wrapper                                 #
################################################################################

class BreakoutEnvWrapper():

    def __init__(self,
                 env,
                 **kw_args):

        super(BreakoutEnvWrapper, self).__init__(env, **kw_args)

        if "Breakout" not in env.spec.id:
            msg  = "ERROR: expected env to be a variation of Breakout "
            msg += "but received {}".format(env.spec.id)
            sys.stderr.write(msg)
            comm.Abort()

        #
        # Breakout doesn't auto-launch the ball, which is a bit of a pain.
        # I don't care to teach the model that it needs to launch the ball
        # itself, so let's launch it autmatically when we reset. Also, let's
        # change the action space to only be (no-op, left, right) since we're
        # removing the ball launch action.
        #
        self.action_space = Discrete(3)

        self.action_map = [0, 2, 3]
        self.cur_lives  = self.env.ale.lives()

    def _set_random_start_pos(self):
        #
        # 20 steps in either direction should be enough to
        # reach either wall.
        #
        rand_step   = np.random.randint(2, 4)
        rand_repeat = np.random.randint(1, 20)

        #
        # NOTE: this calls the environments step function, not our
        # own. This means that we will be skipping some frames and
        # not storing them in memory. This allows for a random
        # starting point when entering a new game.
        #
        for _ in range(rand_repeat):
            self.env.step(rand_step)

    def fire_ball(self):
        return self.env.step(1)

    def true_done_reset(self):
        """
        The action we should take when we are truly done, not just
        when we've acceptable lost a life. This will be called by
        the state_dependent_reset parent method.
        """
        self.env.reset()
        self._set_random_start_pos()
        obs, _, _, _, _ = self.fire_ball()
        return obs, {}

    def false_done_reset(self):
        """
        What to do when we've acceptably lost a life, but we haven't
        ended the game.

        In this case, we need to fire the ball again.
        """
        obs, _, _, _, _ = self.fire_ball()
        return obs, {}


class BreakoutRAMEnvWrapper(BreakoutEnvWrapper, RAMHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size       = 2,
                 allow_life_loss = False,
                 punish_end      = False,
                 skip_k_frames   = 1,
                 **kw_args):

        super(BreakoutRAMEnvWrapper, self).__init__(
            env             = env,
            hist_size       = hist_size,
            allow_life_loss = allow_life_loss,
            skip_k_frames   = skip_k_frames,
            **kw_args)

        self.punish_end = punish_end

    def step(self, action):
        action    = self.action_map[action]
        step_func = lambda a : RAMHistEnvWrapper._env_step(self, a)

        obs, reward, terminated, truncated, info = self._frame_skip_step(
            action    = action,
            step_func = step_func)

        if self.punish_end:
            #
            # Return a negative reward for failure.
            #
            if terminated and reward == 0:
                reward = -1.

        return obs, reward, terminated, truncated, info


class BreakoutPixelsEnvWrapper(BreakoutEnvWrapper, PixelHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size       = 2,
                 allow_life_loss = False,
                 punish_end      = False,
                 skip_k_frames   = 1,
                 **kw_args):


        super(BreakoutPixelsEnvWrapper, self).__init__(
            env             = env,
            hist_size       = hist_size,
            allow_life_loss = allow_life_loss,
            skip_k_frames   = skip_k_frames,
            **kw_args)

        self.punish_end    = punish_end

        #
        # Crop the images by removing the "score" information.
        #
        prev_shape   = env.observation_space.shape
        self.h_start = 20
        self.h_stop  = prev_shape[0]
        self.w_start = 0
        self.w_stop  = prev_shape[1]

        new_shape = (hist_size, self.frame_size, self.frame_size)
        low       = np.zeros(new_shape)
        high      = np.full(new_shape, 255)

        self.observation_space = Box(
            low   = low,
            high  = high,
            shape = new_shape,
            dtype = np.uint8)

    def step(self, action):

        action    = self.action_map[action]
        step_func = lambda a : PixelHistEnvWrapper._env_step(self, a)

        obs, reward, terminated, truncated, info = self._frame_skip_step(
            action    = action,
            step_func = step_func)

        if self.punish_end:
            #
            # Return a negative reward for failure.
            #
            if terminated and reward == 0:
                reward = -1.

        return obs, reward, terminated, truncated, info
