import numpy as np
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class CallableExtent():
    """
        A simple placeholder for a scheduler.
    """

    def __init__(self, val):
        self.val = val

    def finalize(self, *args, **kw_args):
        pass

    def __call__(self, *args, **kw_args):
        return self.val

class StatusScheduler(object):

    def __init__(self,
                 status_key,
                 status_max):

        self.status_max = status_max
        self.status_key = status_key
        self.finalized  = False

    def finalize(self, status_dict):
        """
        """
        self.status_dict = status_dict
        self._validate()
        self.finalized = True

    def _validate(self):
        """
        """
        if self.status_key == "":
            self.finalized = True
            return

        if self.status_key not in self.status_dict["general"]:
            msg  = "ERROR: status_key must exist in status_dict['general']. "
            msg += f"Available keys: {self.status_dict['general'].keys()}."
            rank_print(msg)
            comm.Abort()

        try:
            float(self.status_dict["general"][self.status_key])
        except ValueError:
            msg  = "ERROR: the value for a mapper must be a number!"
            rank_print(msg)
            comm.Abort()

        self.finalized = True

    def _get_step(self):
        assert self.finalized

        if self.status_key == "":
            return 0

        return self.status_dict["general"][self.status_key]

    def __call__(self, iteration, timestep, **kw_args):
        raise NotImplementedError


class LogScheduler(StatusScheduler):

    def __init__(self,
                 status_key,
                 status_max,
                 max_value,
                 min_value,
                 **kw_args):

        super(LogScheduler, self).__init__(
            status_max  = status_max,
            status_key  = status_key,
            **kw_args)

        self.min_value = min_value
        self.max_value = max_value
        self.numerator = np.log(self.status_max) / (max_value - min_value)

    def __call__(self):

        step      = self._get_step()
        dec_value = self.max_value - (np.log(step) / self.numerator)
        dec_value = min(dec_value, self.max_value) 

        return max(dec_value, self.min_value)


class LinearScheduler(StatusScheduler):

    def __init__(self,
                 status_key,
                 status_max,
                 max_value,
                 min_value,
                 **kw_args):

        super(LinearScheduler, self).__init__(
            status_key  = status_key,
            status_max  = status_max,
            **kw_args)

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self):

        step    = self._get_step()
        new_val = self.max_value - (step *
            ((self.max_value - self.min_value) / self.status_max))

        new_val = max(new_val, self.min_value)
        return new_val


class LinearStepScheduler(StatusScheduler):

    def __init__(self,
                 status_key,
                 steps,
                 step_values,
                 ending_value,
                 **kw_args):
        """
            A class that maps status dict entries to new values. Steps should
            be a list containing numeric triggers to look for in the status dict
            in ascending order. As long as
            our step is < the current index of the step list (starting
            at index 0), the associated value from step_values will be returned.
            Once our step exceeds the current step, the index is
            incremented, and the process repeats. If our step ever exceeds
            the last step in steps, ending_value will be returned thereafter.

            Arguments:
                steps         A list of triggers from the status dict.
                step_values   The values corresponding to steps.
                ending_value  The value to use if our step ever exceeds the
                              last entry of steps.
        """
        super(LinearStepScheduler, self).__init__(
            status_max  = 1,
            status_key  = status_key,
            **kw_args)

        self.steps        = steps
        self.step_values  = step_values
        self.ending_value = ending_value
        self.range_idx    = 0

        if len(self.steps) == 0:
            msg = "ERROR: LinearStepScheduler requires at least one step."
            rank_print(msg)
            comm.Abort()

        if len(self.steps) != len(self.step_values):
            msg  = "ERROR: steps and step_values must contain "
            msg += "the same number of entries."
            rank_print(msg)
            comm.Abort()

    def __call__(self):

        step = self._get_step()

        if self.range_idx >= len(self.steps):
            return self.ending_value

        while step > self.steps[self.range_idx]:
            self.range_idx += 1

            if self.range_idx >= len(self.steps):
                return self.ending_value

        return self.step_values[self.range_idx]
