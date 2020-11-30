# imports

import numpy as np

# functions

def stability_cut(f, channel, idx_startwith=None):
    """
    Return all event indices, that are between two stable testpulses

    :param f: handle of the h5 file
    :type f: hdf5 filestream
    :param channel: the channel of which the cut should be done
    :type channel: int
    :param idx: only indices that are in this list can be in the output list
    :type idx: List of ints or None
    :return: the indices of events that are between two stable test pulses
    :rtype: list of ints
    """

    tpas = f['testpulses']['testpulseamplitude']
    tphs = f['testpulses']['mainpar'][channel, :, 0]  # 0 is the mainpar index for pulseheight
    hours_tp = f['testpulses']['hours']
    hours_ev = f['events']['hours']
    idx = np.array(range(len(tpas)))

    unique_tpas = np.unique(tpas)

    # cleaning
    cond = tphs > 0

    for val in unique_tpas:
        # get all hours that are not within 2 standard deviations
        mu = np.mean(tphs[tpas == val])
        sigma = np.std(tphs[tpas == val])

        cond = np.logical_and(cond, np.abs(tphs - mu) < 2*sigma)

    # make the exclusion intervalls
    exclusion = []
    for i, bool in enumerate(cond):
        if bool:
            exclusion.append([hours_tp[i-1], hours_tp[i+1]])

    # both testpulses before and after must have condition true
    ev_cond = [True for i in idx]
    for lb, up in exclusion:
        ev_cond =  np.logical_and(ev_cond, np.logical_and((hours_ev > lb), (hours_ev < up)))

    if idx_startwith is not None:
        ev_cond = np.logical_and(ev_cond, np.in1d(idx, idx_startwith))

    return idx[ev_cond]