import numpy as np
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class CallableValue():
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
                 status_preface = "general"):

        self.status_key     = status_key
        self.finalized      = False
        self.status_preface = status_preface

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

        if self.status_key not in self.status_dict[self.status_preface]:
            msg  = f"ERROR: status_key {self.status_key} does not exists in "
            msg += "status_dict[status_preface]. "
            msg += f"Available keys in self.status_dict['{self.status_preface}']:"
            msg += f"{self.status_dict[self.status_preface].keys()}."
            rank_print(msg)
            comm.Abort()

        try:
            float(self.status_dict[self.status_preface][self.status_key])
        except ValueError:
            msg  = "ERROR: the value for a mapper must be a number!"
            rank_print(msg)
            comm.Abort()

        self.finalized = True

    def _get_step(self):
        assert self.finalized

        if self.status_key == "":
            return 0

        return self.status_dict[self.status_preface][self.status_key]

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
            status_key  = status_key,
            **kw_args)

        self.status_max = status_max
        self.min_value  = min_value
        self.max_value  = max_value
        self.numerator  = np.log(self.status_max) / (max_value - min_value)

    def __call__(self):

        step  = self._get_step()
        value = self.max_value - (np.log(step) / self.numerator)
        value = min(value, self.max_value)

        return max(value, self.min_value)


class LinearScheduler(StatusScheduler):

    def __init__(self,
                 status_key,
                 status_max,
                 max_value,
                 min_value,
                 **kw_args):

        super(LinearScheduler, self).__init__(
            status_key  = status_key,
            **kw_args)

        self.status_max = status_max
        self.min_value  = min_value
        self.max_value  = max_value

    def __call__(self):

        step    = self._get_step()
        new_val = self.max_value - (step *
            ((self.max_value - self.min_value) / self.status_max))

        new_val = max(new_val, self.min_value)
        return new_val


class LinearStepScheduler(StatusScheduler):

    def __init__(self,
                 initial_value,
                 status_key,
                 status_triggers,
                 step_values,
                 compare_fn = np.greater,
                 **kw_args):
        """
            A class that maps status dict entries to scheduled values.
            At each step,
            compare_fn(status_dict[status_key], status_triggers[idx]) will
            be evaluated. The initial_value will be returned at each call until
            the first True evaluation of compare_fn. At that point, each call
            will begin returning entries of step_values, increasing the return
            index for step_values whenever the comparison is evaluated as True.

            Arguments:
                status_key      The status dict key mapping to the value we
                                wish to use as a trigger.
                initial_value   The initial value to return.
                status_triggers A list of triggers from the status dict.
                step_values     The values corresponding to status_triggers.
                compare_fn      The comparison function to use.
        """
        super(LinearStepScheduler, self).__init__(
            status_key = status_key,
            **kw_args)

        self.status_triggers = status_triggers
        self.initial_value   = initial_value
        self.step_values     = step_values
        self.max_idx         = len(self.step_values) - 1
        self.range_idx       = -1
        self.compare_fn      = compare_fn

        if len(self.status_triggers) == 0:
            msg  = "ERROR: LinearStepScheduler requires at least one "
            msg += "status trigger."
            rank_print(msg)
            comm.Abort()

        if len(self.status_triggers) != len(self.step_values):
            msg  = "ERROR: status_triggers and step_values must contain "
            msg += "the same number of entries."
            rank_print(msg)
            comm.Abort()

    def __call__(self):

        #
        # Tricky business: on the first iteration, out status dict won't
        # have been updated yet, so the values are nonsense.
        #
        if self.status_dict["general"]["iteration"] == 0:
            return self.initial_value

        step = self._get_step()

        while (self.range_idx < self.max_idx and
            self.compare_fn(step, self.status_triggers[self.range_idx + 1])):

            self.range_idx = min(self.range_idx + 1, self.max_idx)

        if self.range_idx < 0:
            return self.initial_value

        return self.step_values[self.range_idx]


class ChangeInStateScheduler(StatusScheduler):

    def __init__(self,
                 status_key,
                 compare_fn = np.not_equal,
                 persistent = False,
                 **kw_args):
        """
            A class that tracks changes in a particular status and returns
            compare_fn(prev_status, current_status).
            When persistent is False, the status will be cached every
            iteration. When persistent is True, the cached status will
            only be updated when compare_fn(cache, current) evaluates to True.

            Arguments:
                status_key      The status dict key mapping to the value we
                                wish to track.
                compare_fn      The comparison function to use.
        """

        super(ChangeInStateScheduler, self).__init__(
            status_key = status_key,
            **kw_args)

        self.compare_fn  = compare_fn
        self.prev_status = None
        self.persistent  = persistent

    def __call__(self):

        step = self._get_step()

        if self.prev_status is None:
            self.prev_status = step
            return False

        stat_change = self.compare_fn(step, self.prev_status)

        if self.persistent:
            if stat_change:
                self.prev_status = step
        else:
            self.prev_status = step

        return stat_change
