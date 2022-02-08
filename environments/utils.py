"""
    A home for environment utilities.
"""
import numpy as np

class CustomObservationSpace(object):
    """
        A very simplified observation space.
    """

    def __init__(self,
                 shape):
        """
            Arguments:
                shape    The observation space shape.
        """
        self.shape = shape

class CustomActionSpace(object):
    """
        A very simplified action space.
    """

    def __init__(self,
                 dtype,
                 dims):
        """
            Arguments:
                dtype    The numpy data type of the actions.
                dims     The dimensions of the actions.
        """

        self.dtype = dtype

        if np.issubdtype(self.dtype, np.floating):
            self.shape = dims
        elif np.issubdtype(self.dtype, np.integer):
            self.n = dims
