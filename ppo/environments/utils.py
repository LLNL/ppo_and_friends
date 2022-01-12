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
 
