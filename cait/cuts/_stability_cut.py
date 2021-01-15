# imports

import numpy as np

# functions

def stability_cut(tpas, tphs, hours_tp, hours_ev, significance=3, noise_level=0.005):
    """
    Return all event indices, that are between two stable testpulses
    TODO

    :param f: handle of the h5 file
    :type f: hdf5 filestream
    :param channel: the channel of which the cut should be done
    :type channel: int
    :param idx: only indices that are in this list can be in the output list
    :type idx: List of ints or None
    :return: the indices of events that are between two stable test pulses
    :rtype: list of ints
    """

    unique_tpas = np.unique(tpas)

    # cleaning
    cond = tphs > noise_level

    for val in unique_tpas:
        # get all hours that are not within significance standard deviations
        mu = np.mean(tphs[cond][tpas == val])
        sigma = np.std(tphs[cond][tpas == val])

        cond = np.logical_and(cond, np.abs(tphs - mu) < significance * sigma)

    cond[0] = True  # such that we do not excees boundaries below
    cond[-1] = True

    # make the exclusion intervalls
    exclusion = []
    for i, bool in enumerate(cond):
        if not bool:
            exclusion.append([hours_tp[i-1], hours_tp[i+1]])

    # both testpulses before and after must have condition true
    flag = np.ones(hours_ev, dtype=bool)
    for lb, up in exclusion:
        flag = np.logical_and(flag, np.logical_and((hours_ev >= lb), (hours_ev <= up)))

    return flag