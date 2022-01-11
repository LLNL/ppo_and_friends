import numpy as np

class Decrementer(object):

    def decrement(self, value, iteration, **kw_args):
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

    def decrement(self,
                  iteration,
                  **kw_args):

        dec_value = self.max_value - (np.log(iteration) / self.numerator)
        dec_value = min(dec_value, self.max_value) 
        return max(dec_value, self.min_value)

