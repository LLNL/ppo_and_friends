from mpi4py import MPI
import torch
import numpy as np
import sys

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def rank_print(msg,
               root = 0):
    """
        Print from a single rank.

        Arguments: 
            msg            The message to print.
            root           The rank to print from.
    """
    if root == rank:
        rank_msg = "{}: {}".format(root, msg)
        print(rank_msg)
    sys.stdout.flush()

def set_torch_threads():
    """
        It looks like torch can get greedy with threads at times, and we
        need to keep it in check.

        NOTE: this was taken directly from spinningup with little change.
    """
    if torch.get_num_threads() == 1:
          return

    fair_num_threads = max(int(torch.get_num_threads() / num_procs), 1)
    torch.set_num_threads(fair_num_threads)

def sync_model_parameters(model):
    """
        This allows us to sync all model parameters to be exactly
        like the 0th processor.

        Arguments:
            model    The model whose parameters should be synced.
    """
    if num_procs == 1:
        return

    for param in model.parameters():
        param_data = param.data.numpy()
        comm.Bcast(param_data, root = 0)

def mpi_avg(data):
    """
        Simple function for averaging data across MPI procs.

        Arguments:
            data    The data to average.

        Returns:
            The data averaged across all processors.
    """
    data_type = type(data)

    if (not issubclass(data_type, int) and
        not issubclass(data_type, float) and
        not issubclass(data_type, np.ndarray)):

        msg  = "ERROR: mpi_avg requires input to be of type "
        msg += "float, int, or numpy ndarray."
        print(msg)
        comm.Abort()

    return comm.allreduce(data, MPI.SUM) / num_procs


def mpi_avg_gradients(model):
    """
        Set the gradient buffers of a model to be the average of all
        gradients across each processor.

        Arguments:
            model     The model whose gradients need averaging.
    """
    if num_procs == 1:
        return

    for param in model.parameters():
        param_grad    = param.grad.numpy()
        avg_grad      = mpi_avg(param_grad)
        param_grad[:] = avg_grad[:]
