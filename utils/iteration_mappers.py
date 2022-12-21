import numpy as np
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class StepMapper(object):

    def __init__(self,
                 max_iteration = None,
                 max_timestep  = None):

        if max_iteration == None and max_timestep == None:
            msg  = "ERROR: the chosen step mapper requires that one of "
            msg += "the following options are set: max_iteration, max_timestep."
            rank_print(msg)
            comm.Abort()

        elif max_iteration != None and max_timestep != None:
            msg  = "ERROR: the chosen step mapper requires that either "
            msg += "max_iteration or max_timestep are set, but not both."
            rank_print(msg)
            comm.Abort()

        self.max_iteration = max_iteration
        self.max_timestep  = max_timestep
        self.step_type     = ""

        if max_iteration != None:
            self.step_type = "iteration"
            self.max_step = max_iteration

        elif max_timestep != None:
            self.step_type = "timestep"
            self.max_step = max_timestep

    def _get_step(self,
                  iteration,
                  timestep):

        if self.step_type == "iteration":
            return iteration

        if self.step_type == "timestep":
            return timestep

        rank_print("ERROR: unkonwn step type {}".format(self.step_type))

    def __call__(self, iteration, timestep, **kw_args):
        raise NotImplementedError


class LogDecrementer(StepMapper):

    def __init__(self,
                 max_value,
                 min_value,
                 max_iteration = None,
                 max_timestep  = None,
                 **kw_args):

        super(LogDecrementer, self).__init__(
            max_iteration = max_iteration,
            max_timestep  = max_timestep,
            **kw_args)

        self.min_value = min_value
        self.max_value = max_value
        self.numerator = np.log(self.max_step) / (max_value - min_value)

    def __call__(self,
                 **kw_args):

        step = self._get_step(**kw_args)

        dec_value = self.max_value - (np.log(step) / self.numerator)
        dec_value = min(dec_value, self.max_value) 
        return max(dec_value, self.min_value)


class LinearDecrementer(StepMapper):

    def __init__(self,
                 max_value,
                 min_value,
                 max_iteration = None,
                 max_timestep  = None,
                 **kw_args):

        super(LinearDecrementer, self).__init__(
            max_iteration = max_iteration,
            max_timestep  = max_timestep,
            **kw_args)

        self.min_value     = min_value
        self.max_value     = max_value

    def __call__(self,
                 **kw_args):

        step    = self._get_step(**kw_args)
        new_val = self.max_value - (step *
            ((self.max_value - self.min_value) / self.max_step))

        new_val = max(new_val, self.min_value)
        return new_val


class LinearStepMapper(StepMapper):

    def __init__(self,
                 steps,
                 step_values,
                 ending_value,
                 step_type,
                 **kw_args):
        """
            A class that maps iterations or timeteps to values. Steps should
            be a list containing iterations or timesteps in ascending order. As
            long as our step is < the current index of the step list (starting
            at index 0), the associated value from step_values will be returned.
            Once our step exceeds the current step, the index is
            incremented, and the process repeats. If our step ever exceeds
            the last step in steps, ending_value will be returned thereafter.

            Arguments:
                steps         A list of iterations or timesteps.
                step_values   The values corresponding to steps.
                ending_value  The value to use if our step ever exceeds the
                              last entry of steps.
                step_type     A string determing which step type to use. Options
                              are "iteration" and "timestep".
        """

        avail_types = ["iteration", "timestep"]
        if step_type not in avail_types:
            msg  = "ERROR: received {} for step_type, ".format(step_type)
            msg += "but step_type must be one of the "
            msg += "following: {}".format(avail_types)
            rank_print(msg)
            comm.Abort()

        self.step_type    = step_type
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
                 **kw_args):

        step = self._get_step(**kw_args)

        if self.range_idx >= len(self.steps):
            return self.ending_value

        while step > self.steps[self.range_idx]:
            self.range_idx += 1

            if self.range_idx >= len(self.steps):
                return self.ending_value

        return self.step_values[self.range_idx]
