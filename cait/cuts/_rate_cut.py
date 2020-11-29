# imports

import numpy as np

# functions

def rate_cut(f, type, max_rate, min_rate, idx = None):
    """
    Return all index of events that are in hours with regular rate

    :param f: the file with the events to consider
    :type f: h5 filestream
    :param type: choose events, testpulses or noise
    :type type: string
    :param max_rate: hours that have more events than that get thrown
    :type max_rate: int
    :param min_rate: hours that have less events than that get thrown
    :type min_rate: int
    :param idx: an initial list of indices from that we only cut some away
    :type idx: list or array of integers
    :return: indices that have stable rate
    :rtype: list of ints
    """

    hours = f[type]['events']
    all_idx = list(range(len(hours)))

    hours_hist, bins = np.histogram(hours,
                                    bins=np.arange(int(hours[-1]+1)))
    bins = bins[:-1]  # bins array is one longer than hours array

    # get the hours we want to throw
    cond = np.logical_or(hours_hist > max_rate, hours_hist < min_rate)
    bad_hours = bins[cond]

    # throw them from the index list
    cond = np.in1d(np.floor(hours), bad_hours)
    if idx is not None:
        cond = np.logical_and(cond, np.in1d(np.floor(all_idx), idx))

    return all_idx[cond]
