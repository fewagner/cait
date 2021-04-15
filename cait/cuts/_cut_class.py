# imports

import numpy as np


# the class

class LogicalCut:
    """
    TODO

    :param initial_condition:
    :type initial_condition:
    """

    def __init__(self, initial_condition):

        if len(initial_condition.shape) != 1:
            raise KeyError('initial_condition needs to be a 1-dimensional array!')
        if initial_condition.dtype != 'bool':
            raise KeyError('initial_condition needs to be numpy array with dtype bool!')
        self.cut_flag = initial_condition
        self.all_idx = np.arange(len(initial_condition))

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
        self.cut_flag = np.logical_and(self.cut_flag, condition)

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
