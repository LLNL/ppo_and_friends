from matplotlib import animation
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import os

def save_frames_as_gif(
    frames,
    out_path,
    filename = 'test.gif',
    fps      = 50):
    """
        Save numpy frames as a gif.

        Arguments:
            frames       A list of rendered frames as rgb arrays.
            out_path     The path to save the gif to.
            filename     The name of the gif.
            fps          Frames per second.
    """
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(os.path.join(out_path, filename), fps=fps)
