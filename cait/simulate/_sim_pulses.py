import h5py
import numpy as np
from ..fit._templates import pulse_template
from ._sim_bl import simulate_baselines
from scipy.stats import uniform
from ..fit._saturation import scaled_logistic_curve, scale_factor
from ._generate_pm_par import generate_ps_par


def simulate_events(path_h5,
                    type,
                    name_appendix,
                    size,
                    record_length,
                    nmbr_channels,
                    ph_intervals=[(0, 1), (0, 1)],
                    discrete_ph=None,
                    exceptional_sev_naming=None,
                    channels_exceptional_sev=[0],
                    t0_interval=(-20, 20),  # in ms
                    fake_noise=True,
                    use_bl_from_idx=0,
                    take_idx = None,
                    rms_thresholds=[1, 1],
                    lamb=0.01,
                    sample_length=0.04,
                    saturation=False,
                    reuse_bl=False,
                    ps_dev=False):
    """
    Simulates pulses on a noise baseline.

    Options are to take measured noise baselines
    or fake bl, sev of events or testpulses, pulse heights, onset intervals.

    :param path_h5: Path of the dataset on which the sim is based on.
    :type path_h5: string
    :param type: Either events, testpulses or noise - what type
        of event is simulated; if testpulses the other channels than the first will beignored for the PHs.
    :type type: string
    :param size: Size of the simulated dataset, if fake_noise is False it
        must be smaller than the number of noise baselines in the hdf5 file.
    :type size: integer > 0
    :param record_length: Length of the record window in the hdf5 datset.
    :type record_length: integer
    :param nmbr_channels: The number of channels in the hdf5 dataset.
    :type nmbr_channels: integer
    :param ph_intervals: Intervals for the simulated pulse heights,
        with c the nmbr_channels; the heights are samples uniformly from these intervals.
    :type ph_intervals: list of c intervals
    :param discrete_ph: c Lists of values for the pulse heights for the c channels
        or None for ph intervals; the heights are sampled uniformly for the list.
    :type discrete_ph: list of intervals
    :param exceptional_sev_naming: If set, this is full group name in the HDF5 set for the
        sev used for the simulation of events - by setting this, e.g. carrier events can be
        simulated.
    :type exceptional_sev_naming: string
    :param channel_exceptional_sev: The channels for that the exceptional sev is
        used, e.g. if only for phonon channel, choose [0], if for botch phonon and light, choose [0,1].
    :type channel_exceptional_sev: list of ints
    :param t0_interval: Interval (l,u) with l lower bound for onset and u upper bound.
        The onsets are sampled uniformly from the interval. In ms.
    :type t0_interval: tuple
    :param fake_noise: If true use simulated noise baselines, otherwise measured ones.
    :type fake_noise: bool
    :param use_bl_from_idx: The start index of the baselines that are used.
    :type use_bl_from_idx: int
    :param take_idx: The event indices which we want to use for the simulation.
    :type take_idx: list
    :param rms_threshold: Above which value noise baselines are excluded for the
        distribution of polynomial coefficients; also, a cut value for the baselines if not the
        fake ones but the ones from the h5 set are taken.
    :type rms_threshold: float
    :param lamb: The parameter for the bl simulation method.
    :type lamb: float
    :param sample_length: The length in ms of one sample from an event.
    :type sample_length: float
    :param saturation: If True the logistic curve is applied to the pulses.
    :type saturation: bool
    :param reuse_bl: If True the same baselines are used multiple times to have enough of them
        (use this with care to not have identical copies of events).
    :type reuse_bl: bool
    :param ps_dev: If True the pulse shape parameters are modelled with deviations.
    :type ps_dev: bool
    :return: The simulated events,  the true pulse heights, the onsets of the events.
    :rtype: (3D array of size (nmbr channels, size, record_length), the simulated events,
                2D array of size (nmbr channels, size), the true pulse heights,
                1D array (size), the onsets of the events)
    """
    with h5py.File(path_h5, 'r') as h5f:
        t = (np.arange(0, record_length, dtype=float) - record_length / 4) * sample_length
        nmbr_thrown = 0

        # get baselines
        print('Get Baselines.')
        if fake_noise:
            sim_events, _ = simulate_baselines(path_h5=path_h5,
                                               size=size,
                                               rms_thresholds=rms_thresholds,
                                               lamb=lamb,
                                               verb=True)

            hours, time_s, time_mus = None, None, None

        else:
            if take_idx is None:
                take_idx = []
                if not reuse_bl:
                    if use_bl_from_idx + size <= len(h5f['noise']['event'][0]):
                        bl_rms = np.array(h5f['noise']['fit_rms'][:, use_bl_from_idx:])
                        counter = 0
                        while len(take_idx) < size:  # clean the baselines
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
                        hours = np.array(h5f['noise']['hours'][take_idx])
                        time_s = np.array(h5f['noise']['time_s'][take_idx])
                        time_mus = np.array(h5f['noise']['time_mus'][take_idx])

                    else:
                        raise KeyError('Size must not exceed number of noise bl in hdf5 file!')
                else:  # do reuse baselines
                    reuse_counter = 0
                    bl_rms = np.array(h5f['noise']['fit_rms'])
                    nmbr_bl_total = len(bl_rms[0])
                    counter = 0
                    stop_condition = 0
                    idx_lists = []
                    while stop_condition < size:
                        take_it = True
                        for c in range(nmbr_channels):
                            if (bl_rms[c, counter] > rms_thresholds[c]):  # check rms threshold
                                take_it = False

                        if take_it:
                            take_idx.append(counter)
                            stop_condition += 1

                        counter += 1
                        if counter >= nmbr_bl_total:
                            print('Nmbr resets to start of Baseline dataset: {}'.format(reuse_counter))
                            reuse_counter += 1
                            counter = 0
                            idx_lists.append(np.array(take_idx))
                            take_idx = []
                    idx_lists.append(take_idx)

                    sim_events = np.concatenate([np.array(h5f['noise']['event'][:, il, :]) for il in idx_lists], axis=1)
                    hours = np.concatenate([np.array(h5f['noise']['hours'][il]) for il in idx_lists], axis=0)
                    time_s = np.concatenate([np.array(h5f['noise']['time_s'][il]) for il in idx_lists], axis=0)
                    time_mus = np.concatenate([np.array(h5f['noise']['time_mus'][il]) for il in idx_lists], axis=0)
            else:
                if size > len(take_idx):
                    if reuse_bl:
                        take_idx = np.tile(take_idx, int(size/len(take_idx) + 1))
                    else:
                        raise KeyError('The size is larger than the number of cleaned baselines. Reduce size, '
                                       'provide more baselines or activate reuse_baselines!')
                take_idx = take_idx[:size]
                sim_events = np.array(h5f['noise']['event'][:, take_idx, :])
                hours = np.array(h5f['noise']['hours'][take_idx])
                time_s = np.array(h5f['noise']['time_s'][take_idx])
                time_mus = np.array(h5f['noise']['time_mus'][take_idx])

        # get pulse heights
        print('Get Pulse Heights.')
        phs = np.zeros((nmbr_channels, size))

        if discrete_ph is None:
            randvars = uniform.rvs(size=size)
            for c in range(nmbr_channels):
                phs[c] = (ph_intervals[c][1] - ph_intervals[c][0])*randvars + ph_intervals[c][0]
        else:
            nmbr_phs = len(discrete_ph[0])
            if all(len(discrete_ph[c]) == nmbr_phs for c in range(1, nmbr_channels)):
                randvals = np.random.randint(0, nmbr_phs, size=size)
                for c in range(nmbr_channels):
                    phs[c] = np.array(discrete_ph[c])[randvals]
            else:
                for c in range(nmbr_channels):
                    phs[c] = np.random.choice(discrete_ph[c], size)

        # for c in range(nmbr_channels):
        #     if discrete_ph is None:
        #         phs[c] = uniform.rvs(size=size, loc=ph_intervals[c][0], scale=ph_intervals[c][1] - ph_intervals[c][0])
        #     else:
        #         phs[c] = np.random.choice(discrete_ph[c], size)

        # get t0's
        t0s = uniform.rvs(loc=t0_interval[0], scale=t0_interval[1] - t0_interval[0], size=size)

        # add pulses
        print('Add Pulses to Baselines.')
        if type == 'events':
            used_exept_sevs = 0  # this counts to get the exceptional event at correct index
            for c in range(nmbr_channels):
                if exceptional_sev_naming is None:
                    par = h5f['stdevent' + name_appendix]['fitpar'][c]
                else:
                    if c in channels_exceptional_sev:
                        if len(channels_exceptional_sev) == 1:
                            par = h5f[exceptional_sev_naming]['fitpar']  # has no channels
                        else:
                            par = h5f[exceptional_sev_naming]['fitpar'][used_exept_sevs]
                            used_exept_sevs += 1
                    else:
                        par = h5f['stdevent' + name_appendix]['fitpar'][c]
                for e in range(size):
                    if ps_dev and c == 0:  # so far this only works for the phonon channel
                        if phs[c, e] != 0:
                            par = generate_ps_par(phs[c, e].reshape([-1]))
                            par[0] += t0s[e]
                            pulse = pulse_template(t, *par)
                            pulse = pulse/np.max(pulse)
                            sim_events[c, e] += phs[c, e] * pulse
                    else:
                        use_par = np.copy(par)
                        use_par[0] += t0s[e]
                        sim_events[c, e] += phs[c, e] * pulse_template(t, *use_par)

        elif type == 'testpulses':
            one_true_ph = phs[0]
            if saturation:
                one_true_ph /= scale_factor(*h5f['saturation']['fitpar'][0])
            for c in range(nmbr_channels):
                phs[c] = one_true_ph
                if saturation:
                    log_fitpar = h5f['saturation']['fitpar'][c]
                    phs[c] *= scale_factor(*log_fitpar)
                for e in range(size):
                    par = h5f['stdevent_tp']['fitpar'][c]
                    sim_events[c, e] += phs[c, e] * pulse_template(t + t0s[e], *par)

        elif type == 'noise':
            pass
        else:
            raise KeyError('type must be events, testpulses or noise!')

        # add saturation
        if saturation:
            for c in range(nmbr_channels):
                log_fitpar = h5f['saturation']['fitpar'][c]
                for e in range(size):
                    event = sim_events[c, e]
                    offset = np.mean(event[:int(record_length / 8)])
                    ev_no_offset = event - offset
                    ev_sat = scaled_logistic_curve(ev_no_offset, *log_fitpar)
                    sim_events[c, e] = ev_sat + offset

    return sim_events, phs, t0s, nmbr_thrown, hours, time_s, time_mus
