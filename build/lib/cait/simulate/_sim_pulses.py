import h5py
import numpy as np
from ..fit._templates import pulse_template
from ._sim_bl import simulate_baselines
from scipy.stats import uniform


def simulate_events(path_h5,
                    type,
                    size,
                    record_length,
                    nmbr_channels,
                    ph_intervals=[[0, 1], [0, 1]],
                    discrete_ph=None,
                    t0_interval=[-20, 20],  # in ms
                    fake_noise=True,
                    use_bl_from_idx=0,
                    rms_thresholds=[1, 1],
                    lamb=0.01,
                    sample_length=0.04):
    """
    Simulates pulses on a noise baseline. Options are to take measured noise baselines
    or fake bl, sev of events or testpulses, pulse heights, onset intervals.

    :param path_h5: string, path of the dataset on which the sim is based on
    :param type: string, either events, testpulses or noise - what type
        of event is simulated
    :param size: integer > 0, size of the simulated dataset, if fake_noise is False it
        must be smaller than the number of noise baselines in the hdf5 file
    :param record_length: integer, length of the record window in the hdf5 datset
    :param nmbr_channels: integer, the number of channels in the hdf5 dataset
    :param ph_intervals: list of c intervals for the simulated pulse heights,
        with c the nmbr_channels; the heights are samples uniformly from these intervals
    :param discrete_ph: c Lists of values for the pulse heights for the c channels
        or None for ph intervals; the heights are sampled uniformly for the list
    :param t0_interval: Interval (l,u) with l lower bound for onset and u upper bound;
        the onsets are sampled uniformly from the interval
    :param fake_noise: bool, if true use simulated noise baselines, otherwise measured ones
    :param use_bl_from_idx: the start index of the baselines that are used
    :param rms_threshold: float, above which value noise baselines are excluded for the
        distribution of polynomial coefficients
    :param lamb: float, the parameter for the bl simulation method
    :param sample_length: float, the length in ms of one sample from an event
    :return: (3D array of size (nmbr channels, size, record_length), the simulated events,
                2D array of size (nmbr channels, size), the true pulse heights,
                1D array (size), the onsets of the events)
    """
    h5f = h5py.File(path_h5, 'r')
    t = (np.arange(0, record_length, dtype=float) - record_length / 4) * sample_length
    nmbr_thrown = 0
    take_idx = []

    # get baselines
    if fake_noise:
        sim_events, _ = simulate_baselines(path_h5=path_h5,
                                           size=size,
                                           rms_thresholds=rms_thresholds,
                                           lamb=lamb,
                                           verb=True)
    else:
        if use_bl_from_idx + size <= len(h5f['noise']['event'][0]):
            bl_rms = np.array(h5f['noise']['fit_rms'][:, use_bl_from_idx:])
            counter = 0
            while len(take_idx) < size: # clean the baselines
                take_it = True
                for c in range(nmbr_channels):
                    if (bl_rms[c, counter] > rms_thresholds[c]):  # check rms threshold
                        take_it = False

                if take_it:
                    take_idx.append(counter)
                else:
                    nmbr_thrown += 1

                counter += 1
            take_idx = np.array(take_idx) + use_bl_from_idx
            sim_events = np.array(h5f['noise']['event'][:, take_idx, :])
        else:
            raise KeyError('Size must not exceed number of noise bl in hdf5 file!')

    # get pulse heights
    phs = np.zeros((nmbr_channels, size))
    for c in range(nmbr_channels):
        if discrete_ph is None:
            phs[c] = uniform.rvs(size=size, loc=ph_intervals[c][0], scale=ph_intervals[c][1] - ph_intervals[c][0])
        else:
            phs[c] = np.random.choice(discrete_ph[0], size)

    # get t0's
    t0s = uniform.rvs(loc=t0_interval[0], scale=t0_interval[1] - t0_interval[0], size=size)

    # add pulses
    if type == 'events':
        for c in range(nmbr_channels):
            par = h5f['stdevent']['fitpar'][0]
            for e in range(size):
                sim_events[c, e] += phs[c, e] * pulse_template(t + t0s[e], *par)
    elif type == 'testpulses':
        for c in range(nmbr_channels):
            par = h5f['stdevent_tp']['fitpar'][0]
            for e in range(size):
                sim_events[c, e] += phs[c, e] * pulse_template(t + t0s[e], *par)
    elif type == 'noise':
        pass
    else:
        raise KeyError('type must be events, testpulses or noise!')

    h5f.close()

    return sim_events, phs, t0s, nmbr_thrown
