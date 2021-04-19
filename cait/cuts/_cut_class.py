# imports

import numpy as np


# the class

class LogicalCut:
    """
    A class for the application of logical cuts to given data.

    :param initial_condition: An initial condition can be applied, e.g. ph > 0.01, where ph is an 1D array. The number
        of events is
    :type initial_condition: 1D bool array or None
    """

    def __init__(self, initial_condition=None):

        if initial_condition is not None:
            if len(initial_condition.shape) != 1:
                raise KeyError('initial_condition needs to be a 1-dimensional array!')
            if initial_condition.dtype != 'bool':
                raise KeyError('initial_condition needs to be numpy array with dtype bool!')
            self.all_idx = np.arange(len(initial_condition))
        else:
            self.all_idx = None
        self.cut_flag = initial_condition


    def __len__(self):
        return len(self.get_idx())

    def add_condition(self, condition):
        """
        TODO

        :param condition:
        :type condition:
        """
        if len(condition.shape) != 1:
            raise KeyError('condition needs to be a 1-dimensional array!')
        if condition.dtype != 'bool':
            raise KeyError('condition needs to be numpy array with dtype bool!')
        if self.cut_flag is not None:
            self.cut_flag = np.logical_and(self.cut_flag, condition)
        else:
            self.cut_flag = condition
        if self.all_idx is None:
            self.all_idx = np.arange(len(condition))

    def force_true(self, idx):
        """
        TODO

        :param idx:
        :type idx:
        """
        self.cut_flag = np.logical_or(self.cut_flag, np.in1d(self.all_idx, idx))

    def force_false(self, idx):
        """
        TODO

        :param idx:
        :type idx:
        """
        self.cut_flag = np.logical_and(self.cut_flag, np.logical_not(np.in1d(self.all_idx, idx)))

    def get_flag(self):
        """
        TODO

        :return:
        :rtype:
        """
        return self.cut_flag

    def get_antiflag(self):
        """
        TODO

        :return:
        :rtype:
        """
        return np.logical_not(self.cut_flag)


    def get_idx(self):
        """
        TODO

        :return:
        :rtype:
        """
        return self.all_idx[self.cut_flag]

    def counts(self):
        """
        TODO

        :return:
        :rtype:
        """
        return len(self.get_idx())
