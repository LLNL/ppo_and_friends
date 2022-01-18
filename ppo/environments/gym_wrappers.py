from .utils import *
import sys
import numpy as np
import torch
import gym
import cv2

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


#FIXME: change to inherit from gym.Wrapper
class AtariEnvWrapper(object):

    def __init__(self,
                 env,
                 allow_life_loss = False,
                 **kwargs):

        super(AtariEnvWrapper, self).__init__(**kwargs)

        self.allow_life_loss   = allow_life_loss
        self.life_count        = env.ale.lives()
        self.env               = env
        self.action_space      = env.action_space
        self.true_done         = True
        self.false_done_action = None
        self.true_done_action  = None

    def _check_game_end(self, done):
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

        if self.false_done_action == None:
            sys.stderr.write("\nERROR: false_done_action must be a function.")
            sys.exit(1)
        elif self.true_done_action == None:
            sys.stderr.write("\nERROR: true_done_action must be a function.")
            sys.exit(1)

        if self.true_done:
            self.true_done = False
            self.env.reset()
            obs = self.true_done_action()
        else:
            obs = self.false_done_action()

        self.lives = self.env.ale.lives()
        return obs


class AtariPixels(AtariEnvWrapper):

    def __init__(self,
                 env,
                 allow_life_loss = False,
                 frame_size = 84,
                 **kwargs):

        super(AtariPixels, self).__init__(
            env,
            allow_life_loss,
            **kwargs)

        prev_shape      = env.observation_space.shape
        self.h_start    = 0
        self.h_stop     = prev_shape[0]
        self.w_start    = 0
        self.w_stop     = prev_shape[1]

        self.frame_size = frame_size

        new_shape    = (1, frame_size, frame_size)
        self.observation_space = CustomObservationSpace(new_shape)

    def rgb_to_gray(self, rgb_frame):
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

        h_start = self.h_start if h_start == None else h_start
        h_stop  = self.h_stop if h_stop == None else h_stop
        w_start = self.w_start if w_start == None else w_start
        w_stop  = self.w_stop if w_stop == None else w_stop

        return frame[:, h_start : h_stop, w_start : w_stop]

    def resize_frame(self,
                     frame,
                     shape  = None,
                     interp = cv2.INTER_AREA):

        #
        # c2v is reversed.
        #
        os_shape = (self.observation_space.shape[2],
                    self.observation_space.shape[1])
        shape = os_shape if shape == None else shape

        return cv2.resize(frame, shape, interpolation = interp)

    def rgb_to_processed_frame(self, rgb_frame):
        new_frame = self.rgb_to_gray(rgb_frame)
        new_frame = self.crop_frame(new_frame)
        new_frame = np.expand_dims(self.resize_frame(new_frame.squeeze()), 0)
        return new_frame


class PixelDifferenceEnvWrapper(AtariPixels):

    def __init__(self,
                 env,
                 allow_life_loss = False,
                 **kwargs):

        super(PixelDifferenceEnvWrapper, self).__init__(
            env             = env,
            allow_life_loss = allow_life_loss,
            **kwargs)

        self.prev_frame   = None
        self.action_space = env.action_space
        self.h_start      = 0

    def reset(self):
        cur_frame = self._state_dependent_reset()
        cur_frame = self.rgb_to_processed_frame(cur_frame)

        self.prev_frame = cur_frame

        return self.prev_frame.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

        cur_frame = self.rgb_to_processed_frame(cur_frame)

        diff_frame      = cur_frame - self.prev_frame
        self.prev_frame = cur_frame.copy()

        done, life_lost   = self._check_game_end(done)
        info["life lost"] = life_lost

        return diff_frame, reward, done, info

    def render(self):
        self.env.render()


class PixelHistEnvWrapper(AtariPixels):

    def __init__(self,
                 env,
                 hist_size         = 2,
                 allow_life_loss   = False,
                 use_frame_pooling = True,
                 **kwargs):

        super(PixelHistEnvWrapper, self).__init__(
            env             = env,
            allow_life_loss = allow_life_loss,
            **kwargs)

        self.frame_cache       = None
        self.prev_frame        = None
        self.action_space      = env.action_space
        self.hist_size         = hist_size
        self.use_frame_pooling = use_frame_pooling

    def reset(self):
        cur_frame = self._state_dependent_reset()
        cur_frame = self.rgb_to_processed_frame(cur_frame)

        self.frame_cache = np.tile(cur_frame, (self.hist_size, 1, 1))
        self.prev_frame  = cur_frame.copy()

        return self.frame_cache.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

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

        done, life_lost   = self._check_game_end(done)
        info["life lost"] = life_lost

        return self.frame_cache, reward, done, info

    def render(self):
        self.env.render()


class RAMHistEnvWrapper(AtariEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 allow_life_loss = False,
                 **kwargs):

        super(RAMHistEnvWrapper, self).__init__(
            env             = env,
            allow_life_loss = allow_life_loss,
            **kwargs)

        ram_shape   = env.observation_space.shape
        cache_shape = (ram_shape[0] * hist_size,)

        self.observation_space = CustomObservationSpace(
            cache_shape)

        self.ram_size           = ram_shape[0]
        self.cache_size         = cache_shape[0]
        self.hist_size          = hist_size
        self.env                = env
        self.ram_cache          = None
        self.action_space       = env.action_space

    def _reset_ram_cache(self,
                         cur_ram):
        self.ram_cache = np.tile(cur_ram, self.hist_size)

    def reset(self):
        cur_ram  = self._state_dependent_reset()
        cur_ram  = cur_ram.astype(np.float32) / 255.
        self._reset_ram_cache(cur_ram)

        return self.ram_cache.copy()

    def step(self, action):
        cur_ram, reward, done, info = self.env.step(action)
        cur_ram  = cur_ram.astype(np.float32) / 255.

        self.ram_cache = np.roll(self.ram_cache, -self.ram_size)

        offset = self.cache_size - self.ram_size
        self.ram_cache[offset :] = cur_ram.copy()

        done, life_lost   = self._check_game_end(done)
        info["life lost"] = life_lost

        return self.ram_cache.copy(), reward, done, info

    def render(self):
        self.env.render()


class BreakoutEnvWrapper():

    def __init__(self,
                 env,
                 **kwargs):

        super(BreakoutEnvWrapper, self).__init__(env, **kwargs)

        if "Breakout" not in env.spec._env_name:
            msg  = "ERROR: expected env to be a variation of Breakout "
            msg += "but received {}".format(env.spec._env_name)
            sys.stderr.write(msg)
            sys.exit(1)

        #
        # Breakout doesn't auto-launch the ball, which is a bit of a pain.
        # I don't care to teach the model that it needs to launch the ball
        # itself, so let's launch it autmatically when we reset. Also, let's
        # change the action space to only be (no-op, left, right) since we're
        # removing the ball launch action.
        #
        self.action_space = CustomActionSpace(
            env.action_space.dtype,
            3)

        self.action_map        = [0, 2, 3]
        self.cur_lives         = self.env.ale.lives()
        self.false_done_action = self.false_done_reset
        self.true_done_action  = self.true_done_reset

    def _set_random_start_pos(self):
        #
        # 20 steps in either direction should be enough to
        # reach either wall.
        #
        rand_step   = np.random.randint(2, 4)
        rand_repeat = np.random.randint(1, 20)

        for _ in range(rand_repeat):
            self.env.step(rand_step)

    def fire_ball(self):
        return self.env.step(1)

    def true_done_reset(self):
        self._set_random_start_pos()
        obs, _, _, _ = self.fire_ball()
        return obs

    def false_done_reset(self):
        obs, _, _, _ = self.fire_ball()
        return obs


class BreakoutRAMEnvWrapper(BreakoutEnvWrapper, RAMHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size       = 2,
                 allow_life_loss = False,
                 punish_end      = False,
                 skip_k_frames   = 1,
                 **kwargs):

        super(BreakoutRAMEnvWrapper, self).__init__(
            env             = env,
            hist_size       = hist_size,
            allow_life_loss = allow_life_loss,
            **kwargs)

        self.punish_end    = punish_end
        self.skip_k_frames = skip_k_frames

    def step(self, action):
        action = self.action_map[action]

        k_reward_sum = 0
        done = False

        for k in range(self.skip_k_frames):
            obs, reward, s_done, info = RAMHistEnvWrapper.step(self, action)

            k_reward_sum += reward
            done = done or s_done

            if not done and self.allow_life_loss and info["life lost"]:
                self.fire_ball()

        reward = k_reward_sum

        if self.punish_end:
            #
            # Return a negative reward for failure.
            #
            if done and reward == 0:
                reward = -1.

        self._post_step()

        return obs, reward, done, info

    def reset(self):
        self.env.reset()

        #
        # First, we need to randomly place the paddle somewhere. This
        # will change where the ball is launched from.
        #
        self._set_random_start_pos()

        #
        # Next, launch the ball.
        #
        cur_ram, _, _, _ = self.fire_ball()

        cur_ram  = cur_ram.astype(np.float32) / 255.
        self._reset_ram_cache(cur_ram)

        self._post_step(True)

        return self.ram_cache.copy()


class BreakoutPixelsEnvWrapper(BreakoutEnvWrapper, PixelHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size       = 2,
                 allow_life_loss = False,
                 punish_end      = False,
                 skip_k_frames   = 1,
                 **kwargs):


        super(BreakoutPixelsEnvWrapper, self).__init__(
            env             = env,
            hist_size       = hist_size,
            allow_life_loss = allow_life_loss,
            **kwargs)

        self.punish_end    = punish_end
        self.skip_k_frames = skip_k_frames

        #
        # Crop the images by removing the "score" information.
        #
        prev_shape   = env.observation_space.shape
        self.h_start = 20
        self.h_stop  = prev_shape[0]
        self.w_start = 0
        self.w_stop  = prev_shape[1]

        new_shape    = (hist_size, self.frame_size, self.frame_size)
        self.observation_space = CustomObservationSpace(new_shape)

    def step(self, action):
        action = self.action_map[action]

        k_reward_sum = 0
        done = False

        for k in range(self.skip_k_frames):
            obs, reward, s_done, info = PixelHistEnvWrapper.step(self, action)

            k_reward_sum += reward
            done = done or s_done

            if not done and self.allow_life_loss and info["life lost"]:
                self.fire_ball()

        reward = k_reward_sum

        if self.punish_end:
            #
            # Return a negative reward for failure.
            #
            if done and reward == 0:
                reward = -1.

        return obs, reward, done, info
