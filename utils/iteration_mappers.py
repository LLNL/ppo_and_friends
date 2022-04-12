import numpy as np
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class IterationMapper(object):

    def __call__(self, value, iteration, **kw_args):
        raise NotImplementedError


class LogDecrementer(IterationMapper):

    def __init__(self,
                 max_iteration,
                 max_value,
                 min_value,
                 **kw_args):

        self.min_value = min_value
        self.max_value = max_value
        self.numerator = np.log(max_iteration) / (max_value - min_value)

    def __call__(self,
                  iteration,
                  **kw_args):

        dec_value = self.max_value - (np.log(iteration) / self.numerator)
        dec_value = min(dec_value, self.max_value) 
        return max(dec_value, self.min_value)


class LinearDecrementer(IterationMapper):

    def __init__(self,
                 max_iteration,
                 max_value,
                 min_value,
                 **kw_args):

        self.min_value     = min_value
        self.max_value     = max_value
        self.max_iteration = max_iteration

    def __call__(self,
                 iteration,
                 **kw_args):

        frac = 1.0 - (iteration - 1.0) / self.max_iteration
        new_val = min(self.max_value, frac * self.max_value)
        new_val = max(new_val, self.min_value)
        return new_val


class LinearStepMapper(IterationMapper):

    def __init__(self,
                 steps,
                 step_values,
                 ending_value,
                 **kw_args):
        """
            A class that maps iteration steps to values. Steps should
            be a list containing iteration steps in ascending order. As long
            as our iteration is < the current index of the step list (starting
            at index 0), the associated value from step_values will be returned.
            Once our iteration exceeds the current step, the index is
            incremented, and the process repeats. If our iteration ever exceeds
            the last step in steps, ending_value will be returned thereafter.

            Arguments:
                steps         A list of iterations.
                step_values   The values to corresponding to iterations below
                              those in steps.
                ending_value  The value to use if our iteration ever exceeds the
                              last entry of steps.
        """

        self.steps        = steps
        self.step_values  = step_values
        self.ending_value = ending_value
        self.range_idx    = 0

        if len(self.steps) == 0:
            msg = "ERROR: RangeMapper requires at least one range."
            rank_print(msg)
            comm.Abort()

        if len(self.steps) != len(self.step_values):
            msg  = "ERROR: steps and range values must contain "
            msg += "the same number of entries."
            rank_print(msg)
            comm.Abort()

    def __call__(self,
                 iteration,
                 **kw_args):

        if self.range_idx >= len(self.steps):
            return self.ending_value

        while iteration > self.steps[self.range_idx]:
            self.range_idx += 1

            if self.range_idx >= len(self.steps):
                return self.ending_value

        return self.step_values[self.range_idx]
