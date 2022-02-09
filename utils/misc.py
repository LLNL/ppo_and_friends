import numpy as np
import torch
from .stats import RunningMeanStd
import os
import pickle

def get_action_type(env):
    if np.issubdtype(env.action_space.dtype, np.floating):
        return "continuous"
    elif np.issubdtype(env.action_space.dtype, np.integer):
        return "discrete"
    return "unknown"


def need_action_squeeze(env):
    if np.issubdtype(env.action_space.dtype, np.floating):
        act_dim = env.action_space.shape[0]
    elif np.issubdtype(env.action_space.dtype, np.integer):
        act_dim = env.action_space.n

    action_type = get_action_type(env)

    #
    # Environments are very inconsistent! We need to check what shape
    # they expect actions to be in.
    #
    need_action_squeeze = True
    if action_type == "continuous":
        try:
            padded_shape = (1, act_dim)
            action = np.random.randint(0, 1, padded_shape)
    
            env.reset()
            env.step(action)
            env.reset()
            need_action_squeeze = False
    
        except:
            action_shape = (act_dim,)
            action = np.random.randint(0, 1, action_shape)
    
            env.reset()
            env.step(action)
            env.reset()
            need_action_squeeze = True

    return need_action_squeeze


def update_optimizer_lr(optim, lr):
    for group in optim.param_groups:
        group['lr'] = lr


class RunningStatNormalizer(object):
    """
        A structure that allows for normalizing and de-normalizing
        data based on running stats.
    """

    def __init__(self,
                 name,
                 device,
                 epsilon = 1e-8):
        """
            Arguments:
                name        The name of the structure (used for saving/loading).
                epsilon     A very small number to help avoid 0 errors.
        """
        self.name          = name
        self.running_stats = RunningMeanStd()
        self.epsilon       = torch.tensor([epsilon]).to(device)

    def normalize(self,
                  data,
                  update_stats = True):
        """
            Normalize incoming data and potential update our stats.

            Arguments:
                data           The data to normalize.
                update_stats   Whether or not to update our runnign stats.

            Returns:
                The normalized data.
        """
        if update_stats:
            self.running_stats.update(data.cpu().numpy())

        mean     = torch.tensor(self.running_stats.mean)
        variance = torch.tensor(self.running_stats.variance)

        data = (data - mean) / torch.sqrt(variance + self.epsilon)

        return data

    def denormalize(self,
                    data):
        """
            Denormalize incoming data.

            Arguments:
                data    The data to denormalize.

            Returns:
                The denormalized data.
        """
        mean     = torch.tensor(self.running_stats.mean)
        variance = torch.tensor(self.running_stats.variance)
        data     = mean + (data * torch.sqrt(variance + self.epsilon))

        return data

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        f_name   = "{}_stats.pkl".format(self.name)
        out_file = os.path.join(path, f_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        f_name  = "{}_stats.pkl".format(self.name)
        in_file = os.path.join(path, f_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)
