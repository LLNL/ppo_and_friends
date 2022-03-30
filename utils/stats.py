import numpy as np
from mpi4py import MPI
from ppo_and_friends.utils.mpi_utils import rank_print

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class RunningMeanStd(object):
    """
        A running mean and std tracker.
        NOTE: This is almost identical to the Stable Baselines'
        implementation.
    """

    def __init__(self,
                 shape       = (),
                 epsilon     = 1e-4):
        """
            Arguments:
                shape        The shape of data to track.
                epsilon      A very small number to help avoid 0 errors.
        """

        self.mean     = np.zeros(shape, dtype=np.float32)
        self.variance = np.ones(shape, dtype=np.float32)
        self.count    = epsilon

    def update(self,
               data,
               gather_stats = True):
        """
            Update the running stats.

            Arguments:
                data          A new batch of data.
                gather_stats  Should we gather the data across processors
                              before computing stats?
        """
        #
        # If we're running with multiple processors, I've found that
        # it's very important to normalize environments across all
        # processors. This does add a bit of overhead, but not too much.
        # NOTE: allgathers can be dangerous with large datasets.
        # If this becomes problematic, we can perform all work on rank
        # 0 and broadcast. That approach is just a bit slower.
        #
        if num_procs > 1 and gather_stats:
            data = comm.allgather(data)
            data = np.concatenate(data)

        batch_mean     = np.mean(data, axis=0)
        batch_variance = np.var(data, axis=0)
        batch_size     = data.shape[0]

        self._integrate_batch_data(
            batch_mean,
            batch_variance,
            batch_size)

    def _integrate_batch_data(self,
                              batch_mean,
                              batch_variance,
                              batch_size):
        """
            Integrate a new batch of data into our running stats.

            Arguments:
                batch_mean        The mean of the batch.
                batch_variance    The variance of the batch.
                batch_size        The size of the batch.
        """
        delta     = batch_mean - self.mean
        new_count = self.count + batch_size

        #
        # Update our mean.
        #
        self.mean = self.mean + (delta * (batch_size / new_count))

        #
        # Update our variance.
        #
        m_a = self.variance * self.count
        m_b = batch_variance * batch_size
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_size / \
            (self.count + batch_size)

        self.variance = m_2 / (self.count + batch_size)

        #
        # Update our count.
        #
        self.count += batch_size
