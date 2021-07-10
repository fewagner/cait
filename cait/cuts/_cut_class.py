# imports

import numpy as np


# the class

class LogicalCut:
    """
    A class for the application of logical cuts to given data.

    :param initial_condition: An initial condition can be applied, e.g. ph > 0.01, where ph is an 1D array. The number
        of events is equal to the length of the array.
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
        Add and condition that is linked with and AND statement to all other conditions.

        :param condition: A condition to be applied to all events, e.g. ph > 0.01.
        :type condition: 1D bool array
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
        Force certain events to true.

        :param idx: All indices that we want to set to true.
        :type idx: 1D int array
        """
        self.cut_flag = np.logical_or(self.cut_flag, np.in1d(self.all_idx, idx))

    def force_false(self, idx):
        """
        Force certain events to false.

        :param idx: All indices that we want to set to false.
        :type idx: 1D int array
        """
        self.cut_flag = np.logical_and(self.cut_flag, np.logical_not(np.in1d(self.all_idx, idx)))

    def get_flag(self):
        """
        Return a bool array of all events, indicating which do survive all cuts.

        :return: Indicates which events do survive all cuts.
        :rtype: 1D int array
        """
        return self.cut_flag

    def get_antiflag(self):
        """
        Return a bool array of all events, indicating which do not survive all cuts.

        :return: Indicates which events do not survive all cuts.
        :rtype: 1D int array
        """
        return np.logical_not(self.cut_flag)


    def get_idx(self):
        """
        Return an int array of all event indices that survive all cuts.

        :return: The indices that survive all cuts.
        :rtype: 1D int array
        """
        return self.all_idx[self.cut_flag]

    def get_antiidx(self):
        """
        Return an int array of all event indices that do not survive all cuts.

        :return: The indices that do not survive all cuts.
        :rtype: 1D int array
        """
        return self.all_idx[np.logical_not(self.cut_flag)]

    def total(self):
        """
        Return the total number of events.

        :return: The total number of events.
        :rtype: int
        """
        return self.get_flag().shape[0]

    def counts(self):
        """
        Return the number of events that survive all cuts.

        :return: The number of events that survive all cuts.
        :rtype: int
        """
        return self.get_idx().shape[0]
