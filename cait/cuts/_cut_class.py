# imports

import numpy as np


# the class

class LogicalCut:
    """
    TODO
    """

    def __init__(self, initial_condition):
        """
        TODO

        :param initial_condition:
        :type initial_condition:
        """
        self.cut_flag = initial_condition
        self.all_idx = np.arange(len(initial_condition))

    def add_condition(self, condition):
        """
        TODO

        :param condition:
        :type condition:
        :return:
        :rtype:
        """
        self.cut_flag = np.logical_and(self.cut_flag, condition)

    def force_true(self, idx):
        """
        TODO

        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        self.cut_flag = np.logical_or(self.cut_flag, np.in1d(self.all_idx, idx))

    def force_false(self, idx):
        """
        TODO

        :param idx:
        :type idx:
        :return:
        :rtype:
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
