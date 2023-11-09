import numpy as np
import os
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI
import yaml

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

    def save_info(self, *args, **kw_args):
        pass

    def load_info(self, *args, **kw_args):
        pass

    def __call__(self, *args, **kw_args):
        return self.val

class FreezeCyclingScheduler():

    def __init__(
        self,
        policy_groups,
        iterations,
        delay   = -1,
        verbose = False):
        """
        A sechduler for "freeze cycling". Freeze cycling is when we
        cycle through freezing all but one group of policies at a time.
        The frozen policies will not be updated while the unfrozen policies
        continue to train and update. Every N iterations, the frozen policies
        are unfrozen, and the previously unfrozen policies are frozen, and this
        cycle continues on indefinitely.

        Each time a policy is frozen, its networks will be saved with
        a tag signifying which iteration it was frozen at.

        Example:
          policy_groups <- [group_1, group_2, group_3] s.t.
          group_1 <- [policy_0, policy_1]
          group_2 <- [policy_2, policy_3]
          group_3 <- [policy_4]

          Let "iterations" <- 10

          When training, we'll see the following behavior.

          itreations 0 -> 9:
              gropu_1: active
              gropu_2: frozen
              gropu_3: frozen

          itreations 10 -> 19:
              gropu_1: frozen
              gropu_2: active
              gropu_3: frozen

          itreations 20 -> 29:
              gropu_1: frozen
              gropu_2: frozen
              gropu_3: active

          itreations 30 -> 39:
              gropu_1: active
              gropu_2: frozen
              gropu_3: frozen

        Parameters:
        -----------
        policy_groups: list
            A list of policy groups. Each element in the list must
            be a list containing policy ids to group together when
            freezing. For example, imagine we have a game with two teams,
            TeamA and TeamB, and 3 policies, policy_0, policy_1, policy_2.
            TeamA consists of policy_0 and policy_1, and TeamB consists of
            policy_2. In this case, we could create a freeze cycle for
            TeamA and TeamB by setting policy_groups to
            [['policy_0', 'policy_1'], ['policy_2']] OR
            [['policy_0', 'policy_1']]. Any policy that doesn't show up
            in policy_groups will not be grouped with another policy.
        iterations: int
            The frequency in iterations with which to cycle frozen policies.
        delay: int
            Delay freezing any policies until the current iteration is
            > delay.
        verbose: bool
            Enable verbosity?
        """
        self.policy_groups = policy_groups
        self.iterations    = iterations
        self.delay         = delay
        self.status_dict   = None
        self.policies      = None
        self.policy_ids    = None
        self.finalized     = False
        self.num_groups    = len(policy_groups)
        self.active_idx    = 0
        self.verbose       = verbose

    def finalize(self, state_path, status_dict, policies):
        """
        Finalize our scheduler.

        Parameters:
        -----------
        state_path: str
            The state path where the policies are being saved.
        status_dict: dict
            A reference to the status dictionary.
        policies: dict
            A reference to the policy dictionary.
        """
        self.state_path  = state_path
        self.status_dict = status_dict
        self.policies    = policies
        self.policy_ids  = tuple(policies.keys())
        self._validate()

    def _validate(self):
        """
        Validate this scheduler.
        """
        #
        # Make sure that all of the user policies are valid.
        #
        for group in self.policy_groups:
            for policy_id in group:
                if policy_id not in self.policies:
                    msg  = f"ERROR: policy {policy_id} from policy group "
                    msg += f"{group} is not a valid policy."
                    rank_print(msg)
                    comm.Abort()

        refined_groups = []
        for group in self.policy_groups:
            refined_groups.append(group)

        for policy_id in self.policies:
            policy_found = False

            for group in self.policy_groups:
                if policy_id in group:
                    policy_found = True
                    break

            #
            # For any policies that aren't defined in policy_groups, add
            # them as their own group.
            #
            if not policy_found:
                refined_groups.append([policy_id])

        self.policy_groups = refined_groups
        self.num_groups    = len(self.policy_groups)
        self.finalized     = True

    def save_info(self):
        """
        Save out any info that's necessary for continuing training.
        """
        info_file = os.path.join(self.state_path, "FreezeCyclingScheduler.yaml")
        info_dict = {}
        info_dict["active_idx"] = self.active_idx

        with open(info_file, "w") as out_f:
            yaml.dump(info_dict, out_f, default_flow_style=False)

    def load_info(self):
        """
        Load any info that's necessary for continuing training.
        """
        info_file = os.path.join(self.state_path, "FreezeCyclingScheduler.yaml")

        if os.path.exists(info_file):
            with open(info_file, "r") as in_f:
                info_dict = yaml.safe_load(in_f)

            self.active_idx = info_dict["active_idx"]

    def _freeze_group(self, group_idx):
        """
        Freeze one of our policy groups.

        Parameters:
        -----------
        group_idx: int
            An index mapping to the policy group to freeze.
        """
        if self.verbose:
            rank_print(f"****Freezing policies: {self.policy_groups[group_idx]}****")

        for policy_id in self.policy_groups[group_idx]:
            self.policies[policy_id].freeze()

            if rank == 0:
                tag = self.status_dict["global status"]["iteration"]
                self.policies[policy_id].save(self.state_path, tag)

    def _unfreeze_group(self, group_idx):
        """
        Un-freeze one of our policy groups.

        Parameters:
        -----------
        group_idx: int
            An index mapping to the policy group to un-freeze.
        """
        if self.verbose:
            rank_print(f"****Un-freezing policies: {self.policy_groups[group_idx]}****")

        for policy_id in self.policy_groups[group_idx]:
            self.policies[policy_id].unfreeze()

    def __call__(self):
        """
        Cycle through freezing groups of policies.
        """
        current_iteration = self.status_dict["global status"]["iteration"]

        #
        # First step is to freeze all but one policy.
        #
        if current_iteration == (self.delay + 1):
            if self.verbose:
                rank_print(f"****Beginning freeze cycling!****")

            for group_idx in range(self.num_groups):
                self._freeze_group(group_idx)

            self._unfreeze_group(self.active_idx)

        #
        # If we're beyond the delay, we can start cycling.
        #
        elif (current_iteration > (self.delay + 1) and
            current_iteration % self.iterations == 0):

            group_to_freeze = self.active_idx
            self.active_idx = (self.active_idx + 1) % self.num_groups

            self._freeze_group(group_to_freeze)
            self._unfreeze_group(self.active_idx)
        

class StatusScheduler():

    def __init__(self,
                 status_key,
                 status_preface = "global status"):

        self.status_key     = status_key
        self.finalized      = False
        self.status_preface = status_preface

    def finalize(self, status_dict):
        """
        Finalize our scheduler.

        Parameters:
        -----------
        status_dict: dict
            A reference to a status dictionary.
        """
        self.status_dict = status_dict
        self._validate()
        self.finalized = True

    def _validate(self):
        """
        Validate this scheduler.
        """
        if self.status_key == "":
            self.finalized = True
            return

        if self.status_key not in self.status_dict[self.status_preface]:
            msg  = f"ERROR: status_key {self.status_key} does not exist in "
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

        new_val = min(max(new_val, self.min_value), self.max_value)
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

        parameters:
        -----------
        status_key: str
            The status dict key mapping to the value we
            wish to use as a trigger.
        initial_value: float or int
            The initial value to return.
        status_triggers: list
            A list of triggers from the status dict.
        step_values: list
            The values corresponding to status_triggers.
        compare_fn: function
            The comparison function to use.
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
        if self.status_dict["global status"]["iteration"] == 0:
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

        Parameters:
        -----------
        status_key: str
            The status dict key mapping to the value we
            wish to track.
        compare_fn: function
            The comparison function to use.
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
