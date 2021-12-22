from .utils import *
import numpy as np
import torch
import gym
import cv2

class CartPoleEnvManager(object):

    def __init__(self):
        super(CartPoleEnvManager, self)

        self.env            = gym.make("CartPole-v0").unwrapped
        self.current_screen = None
        self.done           = False
        self.action_space   = self.env.action_space

        self.env.reset()
        screen_size = self.get_screen_height() * self.get_screen_width() * 3
        self.observation_space = CustomObservationSpace((screen_size,))

    def reset(self):
        self.env.reset()
        self.current_screen = None
        return self.get_screen_state()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def step(self, action):
        _, reward, self.done, info = self.env.step(action.item())
        obs = self.get_screen_state()
        return obs, reward, self.done, info

    def just_starting(self):
        return self.current_screen is None

    def get_screen_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = np.zeros_like(self.current_screen)
            return black_screen.flatten()
        else:
            screen_1 = self.current_screen
            screen_2 = self.get_processed_screen()
            self.current_screen = screen_2
            return (screen_2 - screen_1).flatten()

    def get_screen_height(self):
        return self.get_processed_screen().shape[2]

    def get_screen_width(self):
        return self.get_processed_screen().shape[3]

    def get_processed_screen(self):
        screen = self.render("rgb_array").transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):

        screen_height = screen.shape[1]
        top           = int(screen_height * 0.4)
        bottom        = int(screen_height * 0.8)

        screen_width  = screen.shape[2]
        left          = int(screen_width * 0.1)
        right         = int(screen_width * 0.9)
        screen        = screen[:, top : bottom, left : right]

        return screen

    def transform_screen_data(self, screen):

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)

        resize = t_transforms.Compose([
            t_transforms.ToPILImage(),
            t_transforms.Resize((40, 90)),
            t_transforms.ToTensor()])

        return resize(screen).unsqueeze(0).numpy()


#FIXME: change to inherit from gym.Wrapper
class AtariEnvWrapper(object):

    def __init__(self,
                 env,
                 min_lives = -1,
                 **kwargs):

        super(AtariEnvWrapper, self).__init__(**kwargs)

        self.min_lives    = min_lives
        self.env          = env
        self.action_space = env.action_space

    def _end_game(self, done):
        return done or self.env.ale.lives() < self.min_lives


class AtariPixels(AtariEnvWrapper):

    def __init__(self,
                 env,
                 min_lives  = -1,
                 frame_size = 84,
                 **kwargs):

        super(AtariPixels, self).__init__(
            env,
            min_lives,
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
                 min_lives = -1,
                 **kwargs):

        super(PixelDifferenceEnvWrapper, self).__init__(
            env       = env,
            min_lives = min_lives,
            **kwargs)

        self.prev_frame   = None
        self.action_space = env.action_space
        self.h_start      = 0

    def reset(self):
        cur_frame = self.env.reset()
        cur_frame = self.rgb_to_processed_frame(cur_frame)

        self.prev_frame = cur_frame

        return self.prev_frame.copy()

    def step(self, action):
        cur_frame, reward, done, info = self.env.step(action)

        cur_frame = self.rgb_to_processed_frame(cur_frame)

        diff_frame      = cur_frame - self.prev_frame
        self.prev_frame = cur_frame.copy()

        done = self._end_game(done)

        return diff_frame, reward, done, info

    def render(self):
        self.env.render()


class PixelHistEnvWrapper(AtariPixels):

    def __init__(self,
                 env,
                 hist_size         = 2,
                 min_lives         = -1,
                 use_frame_pooling = True,
                 **kwargs):

        super(PixelHistEnvWrapper, self).__init__(
            env       = env,
            min_lives = min_lives,
            **kwargs)

        self.frame_cache       = None
        self.prev_frame        = None
        self.action_space      = env.action_space
        self.hist_size         = hist_size
        self.use_frame_pooling = use_frame_pooling

    def reset(self):
        cur_frame = self.env.reset()
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

        ###FIXME: remove
        #from PIL import Image
        #import sys
        #first = self.frame_cache[-1].squeeze()
        #first = (first * 255.).astype(np.uint8)
        #img = Image.fromarray(first, 'L')
        #img.show()

        #last = self.frame_cache[0].squeeze()
        #last = (last * 255.).astype(np.uint8)
        #img = Image.fromarray(last, 'L')
        #img.show()

        #input("")
        ###sys.exit(1)

        done = self._end_game(done)

        return self.frame_cache, reward, done, info

    def render(self):
        self.env.render()


class RAMHistEnvWrapper(AtariEnvWrapper):

    def __init__(self,
                 env,
                 hist_size = 2,
                 min_lives = -1,
                 **kwargs):

        super(RAMHistEnvWrapper, self).__init__(
            env       = env,
            min_lives =  min_lives,
            **kwargs)

        ram_shape   = env.observation_space.shape
        cache_shape = (ram_shape[0] * hist_size,)

        self.observation_space = CustomObservationSpace(
            cache_shape)

        self.min_lives          = min_lives
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
        cur_ram  = self.env.reset()
        cur_ram  = cur_ram.astype(np.float32) / 255.
        self._reset_ram_cache(cur_ram)

        return self.ram_cache.copy()

    def step(self, action):
        cur_ram, reward, done, info = self.env.step(action)
        cur_ram  = cur_ram.astype(np.float32) / 255.

        self.ram_cache = np.roll(self.ram_cache, -self.ram_size)

        offset = self.cache_size - self.ram_size
        self.ram_cache[offset :] = cur_ram.copy()

        done = self._end_game(done)

        return self.ram_cache.copy(), reward, done, info

    def render(self):
        self.env.render()


class BreakoutEnvWrapper():

    def __init__(self,
                 env,
                 auto_fire = False,
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

        self.action_map = [0, 2, 3]
        self.auto_fire  = auto_fire
        self.cur_lives  = self.env.ale.lives()

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

    def _post_step(self,
                   reset = False):

        ret = None

        if not reset and self.auto_fire and self.env.ale.lives() < self.lives:
            ret = self.fire_ball()

        self.lives = self.env.ale.lives()

        return ret

class BreakoutRAMEnvWrapper(BreakoutEnvWrapper, RAMHistEnvWrapper):

    def __init__(self,
                 env,
                 hist_size     = 2,
                 min_lives     = -1,
                 punish_end    = False,
                 skip_k_frames = 1,
                 **kwargs):

        super(BreakoutRAMEnvWrapper, self).__init__(
            env       = env,
            hist_size = hist_size,
            min_lives = min_lives,
            **kwargs)

        self.punish_end    = punish_end
        self.skip_k_frames = skip_k_frames

    def step(self, action):
        action = self.action_map[action]

        for k in range(self.skip_k_frames):
            obs, reward, done, info = RAMHistEnvWrapper.step(self, action)

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
                 hist_size     = 2,
                 min_lives     = -1,
                 punish_end    = False,
                 skip_k_frames = 1,
                 **kwargs):


        super(BreakoutPixelsEnvWrapper, self).__init__(
            env       = env,
            hist_size = hist_size,
            min_lives = min_lives,
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

        for k in range(self.skip_k_frames):
            obs, reward, done, info = PixelHistEnvWrapper.step(self, action)

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
        # will change where the ball is launched from. 20 steps in either
        # direction from the default start is enough to get to the wall.
        #
        self._set_random_start_pos()

        #
        # Next, launch the ball.
        #
        cur_frame, _, _, _ = self.fire_ball()

        cur_frame = self.rgb_to_processed_frame(cur_frame)
        self.frame_cache = np.tile(cur_frame, (self.hist_size, 1, 1))
        self.prev_frame  = cur_frame.copy()

        self._post_step(True)

        return self.frame_cache.copy()
