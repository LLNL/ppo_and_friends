import numpy as np

class CustomObservationSpace(object):

    def __init__(self,
                 shape):
        self.shape = shape

class CustomActionSpace(object):

    def __init__(self,
                 dtype,
                 dims):

        self.dtype = dtype

        if np.issubdtype(self.dtype, np.floating):
            self.shape = dims
        elif np.issubdtype(self.dtype, np.integer):
            self.n = dims
 

#TODO: reference stable baselines in doc. This is almost identical.
class RunningMeanStd(object):

    def __init__(self,
                 shape,
                 epsilon = 1e-4):

        self.mean     = np.zeros(shape, dtype=np.float32)
        self.variance = np.ones(shape, dtype=np.float32)
        self.count    = epsilon

    def update(self, data):
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

        delta     = batch_mean - self.mean
        new_count = self.count + batch_size

        #
        # Update our mean.
        #
        self.mean = self.mean + (delta * (batch_size / new_count))

        #
        # Update our variance.
        #
        m_a       = self.variance * self.count
        m_b       = batch_variance * batch_size
        m_2       = m_a + m_b + np.square(delta) * self.count * batch_size / \
            (self.count + batch_size)

        self.variance = m_2 / (self.count + batch_size)

        #
        # Update our count.
        #
        self.count += batch_size
