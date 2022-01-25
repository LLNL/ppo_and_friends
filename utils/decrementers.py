import numpy as np

class Decrementer(object):

    def __call__(self, value, iteration, **kw_args):
        raise NotImplementedError


class LogDecrementer(Decrementer):

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


class LinearDecrementer(Decrementer):

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
