# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from multiprocessing import Pool
from ..features._mp import calc_main_parameters, calc_additional_parameters
from ..features._ph_corr import calc_correlated_ph
from ..filter._of import optimal_transfer_function
from ..fit._sev import generate_standard_event
from ..filter._of import get_amplitudes
from warnings import warn
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import pulse_template
from ..filter._ma import rem_off
from ..trigger._peakdet import get_triggers
from tqdm.auto import trange

from ..data._baselines import calculate_mean_nps


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class FeaturesMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the calculation of features of the data.
    """

    # -----------------------------------------------------------
    # FEATURE CALCULATION
    # -----------------------------------------------------------

    # Calculate MP
    def calc_mp(self, type='events', path_h5=None, processes=4, down=1,
                max_bounds=None):
        """
        Calculate the Main Parameters for the Events in an HDF5 File.

        This method is described in "CRESST Collaboration, First results from the CRESST-III low-mass dark matter program"
        (10.1103/PhysRevD.100.102002).

        :param type: The group in the HDF5 set, either events or testpulses.
        :type type: string
        :param path_h5: An alternative full path to a hdf5 file, e.g. "data/bck_001.h5".
        :type path_h5: string or None
        :param processes: The number of processes to use for the calculation.
        :type processes: int
        :param down: The events get downsampled by this factor for the calculation of main parameters.
        :type down: int
        :param max_bounds: The interval of indices to which we restrict the maximum search for the pulse height.
        :type max_bounds: tuple of two ints
        """

        if not path_h5:
            path_h5 = self.path_h5

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]
            nmbr_ev = events['event'].shape[1]

            print('CALCULATE MAIN PARAMETERS.')

            with Pool(processes) as p:  # basically a for loop running on 4 processes
                mainpar_list_event = []
                for c in range(self.nmbr_channels):
                    mainpar_list_event.append(p.starmap(
                        calc_main_parameters,
                        [(events['event'][c, i, :], down, max_bounds) for i in range(nmbr_ev)]))
            mainpar_event = np.array([[o.getArray() for o in element] for element in mainpar_list_event])

            events.require_dataset(name='mainpar',
                                   shape=(mainpar_event.shape),
                                   dtype='float')
            events['mainpar'][...] = mainpar_event

            events['mainpar'].attrs.create(name='pulse_height', data=0)
            events['mainpar'].attrs.create(name='t_zero', data=1)
            events['mainpar'].attrs.create(name='t_rise', data=2)
            events['mainpar'].attrs.create(name='t_max', data=3)
            events['mainpar'].attrs.create(name='t_decaystart', data=4)
            events['mainpar'].attrs.create(name='t_half', data=5)
            events['mainpar'].attrs.create(name='t_end', data=6)
            events['mainpar'].attrs.create(name='offset', data=7)
            events['mainpar'].attrs.create(name='linear_drift', data=8)
            events['mainpar'].attrs.create(name='quadratic_drift', data=9)

    # calc stdevent testpulses
    def calc_sev(self,
                 type='events',
                 use_labels=False,
                 correct_label=None,
                 use_idx=None,
                 name_appendix='',
                 pulse_height_interval=None,
                 left_right_cutoff=None,
                 rise_time_interval=None,
                 decay_time_interval=None,
                 onset_interval=None,
                 remove_offset=True,
                 baseline_model='constant',
                 verb=True,
                 scale_fit_height=True,
                 scale_to_unit=None,
                 sample_length=None,
                 t0_start=None,
                 opt_start=False,
                 memsafe=True,
                 batch_size=1000,
                 lower_bound_tau=None,
                 upper_bound_tau=None,
                 pretrigger_samples=500,
                 ):
        """
        Calculate the Standard Event for the Events in the HDF5 File.

        This method is described in "CRESST Collaboration, First results from the CRESST-III low-mass dark matter program"
        (10.1103/PhysRevD.100.102002).

        :param type: The group name in the HDF5 set, either "events" or "testpulses".
        :type type: string
        :param use_labels: Tf True a labels file must be included in the hdf5 file,
            then only the events labeled as events or testpulses are included in the calculation.
        :type use_labels: bool
        :param correct_label: The label to be used for the sev generation.
        :type correct_label: int
        :param use_idx: Only these indices are included for the sev generation.
        :type use_idx: list of ints
        :param name_appendix: This gets appended to the group name stdevent in the HDF5 set.
        :type name_appendix: string
        :param pulse_height_interval: The upper and lower bound for the pulse heights to include into the creation
            of the SEV.
        :type pulse_height_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param left_right_cutoff: The maximal abs value of the R-L baseline difference of events
            to be included in the SEV calculation.
        :type left_right_cutoff: list of NMBR_CHANNELS floats
        :param rise_time_interval: The upper
            and lower bound for the rise time in ms to include into the creation of the SEV.
        :type rise_time_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param decay_time_interval: The upper
            and lower bound for the decay time in ms to include into the creation of the SEV.
        :type decay_time_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param onset_interval:  The upper
            and lower bound for the onset time in ms to include into the creation of the SEV.
        :type onset_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param remove_offset: Tf True the offset is removed before the events are superposed for the
            sev calculation. Highly recommended!
        :type remove_offset: bool
        :param baseline_model: Either 'constant', 'linear' or 'exponential'. The baseline model substracted from all
            events.
        :type baseline_model: str
        :param verb: If True some verbal feedback is output about the progress of the method.
        :type verb: bool
        :param scale_fit_height: If True the parametric fit to the sev is normalized to height 1 after
            the fit is done.
        :type scale_fit_height: bool
        :param scale_to_unit: If True corresponding to a channel, the standard event is scaled to 1. Default True. If
            False for a channel, the parametric fit is not applied but automatically set to values that produce an empty array.
            In this case, also the scale_fit_height is not done for this channel.
        :type scale_to_unit: bool list of length nmbr_channels or None
        :param sample_length: The length of one sample in milliseconds. If None, this is calculated from the sample
            frequency.
        :type sample_length: float
        :param t0_start: The start values for t0 in the fit.
        :type t0_start: 2-tupel of floats
        :param opt_start: If true, a pre-fit is applied to find optimal start values.
        :type opt_start: bool
        :param memsafe: Recommended! If activated, not all events get loaded into memory.
        :type memsafe: bool
        :param batch_size: The batch size for the calculation of the SEV.
        :type batch_size: int
        :param lower_bound_tau: The lower bound for all tau values in the fit.
        :type lower_bound_tau: float
        :param upper_bound_tau: The upper bound for all tau values in the fit.
        :type upper_bound_tau: float
        :param pretrigger_samples: The number of samples from start of the record window that are considered the pre
            trigger region.
        :type pretrigger_samples: int
        """

        assert not memsafe or (use_labels == False and correct_label is None and pulse_height_interval is None and
                               left_right_cutoff is None and rise_time_interval is None and decay_time_interval is None
                               and onset_interval is None), \
            'The memsafe option does not allow for correct_label, ' \
            'pulse_height_interval, left_right_cutoff, rise_time_interval,' \
            'decay_time_interval, onset_interval argument. It is ' \
            'recommended to hand a use_labels flag instead!'

        assert memsafe or (lower_bound_tau is None and upper_bound_tau is None and
                           baseline_model == 'constant'), 'For using arguments lower_bound_tau, ' \
                                                          'upper_bound_tau and baseline_model activate memsafe option!'

        if scale_to_unit is None:
            scale_to_unit = [True for i in range(self.nmbr_channels)]

        if sample_length is None:
            sample_length = 1 / self.sample_frequency * 1000

        if lower_bound_tau is None:
            lower_bound_tau = [1e-2 for i in range(self.nmbr_channels)]

        if upper_bound_tau is None:
            upper_bound_tau = [3e3 for i in range(self.nmbr_channels)]

        # set the start values for t0
        if t0_start is None:
            t0_start = [-3 for i in range(self.nmbr_channels)]

        with h5py.File(self.path_h5, 'r+') as h5f:
            events = h5f[type]['event']
            mainpar = h5f[type]['mainpar']

            std_evs = []

            # if no pulse_height_interval is specified set it to average values for all channels
            if pulse_height_interval == None:
                pulse_height_interval = [None
                                         for c in range(self.nmbr_channels)]

            # fix the issue with different arguments for different channels
            inp = [left_right_cutoff, rise_time_interval, decay_time_interval, onset_interval]
            for i, var in enumerate(inp):
                if var is None:
                    inp[i] = [None for c in range(self.nmbr_channels)]

            if use_labels:
                labels = h5f[type]['labels']
            else:
                labels = [None for c in range(self.nmbr_channels)]

            if correct_label is None:
                if type == 'events':
                    sev = h5f.require_group('stdevent' + name_appendix)
                    correct_label = 1
                elif type == 'testpulses':
                    sev = h5f.require_group('stdevent_tp' + name_appendix)
                    correct_label = 2
                else:
                    raise NotImplementedError('Type must be events or testpulses!')
            else:
                sev = h5f.require_group('stdevent' + name_appendix)

            if use_idx is None:
                use_idx = list(range(events.shape[0]))

            for c in range(self.nmbr_channels):
                print('\nCalculating SEV for Channel {}'.format(c))
                if not memsafe:
                    std_evs.append(generate_standard_event(events=events[c, use_idx, :],
                                                           main_parameters=mainpar[c, use_idx, :],
                                                           labels=labels[c],
                                                           correct_label=correct_label,
                                                           pulse_height_interval=pulse_height_interval[c],
                                                           left_right_cutoff=inp[0][c],
                                                           rise_time_interval=inp[1][c],
                                                           decay_time_interval=inp[2][c],
                                                           onset_interval=inp[3][c],
                                                           remove_offset=remove_offset,
                                                           verb=verb,
                                                           scale_fit_height=scale_fit_height,
                                                           scale_to_unit=scale_to_unit[c],
                                                           sample_length=sample_length,
                                                           t0_start=t0_start[c],
                                                           opt_start=opt_start,
                                                           ))
                else:
                    print('{} Events used to generate Standardevent.'.format(len(use_idx)))
                    nmbr_batches = int(len(use_idx) / batch_size)
                    std_evs.append([np.zeros(events.shape[2]), ])

                    for b in range(nmbr_batches):
                        start = int(b * batch_size)
                        stop = int((b + 1) * batch_size)
                        ev = np.array(events[c, use_idx[start:stop]])
                        if remove_offset:
                            rem_off(ev, baseline_model, pretrigger_samples)
                        std_evs[c][0] += np.sum(ev, axis=0)

                    # last batch
                    start = int(nmbr_batches * batch_size)
                    ev = np.array(events[c, use_idx[start:]])
                    if remove_offset:
                        rem_off(ev, baseline_model, pretrigger_samples)
                    std_evs[c][0] += np.sum(ev, axis=0)
                    std_evs[c][0] /= len(use_idx)

                    if scale_to_unit[c]:
                        std_evs[c][0] /= np.max(std_evs[c][0])

                        par = fit_pulse_shape(std_evs[c][0], sample_length=sample_length,
                                              t0_start=t0_start[c],
                                              opt_start=opt_start,
                                              lower_bound_tau=lower_bound_tau[c],
                                              upper_bound_tau=upper_bound_tau[c], )

                        if scale_fit_height:
                            t = (np.arange(0, len(std_evs[c][0]), dtype=float) - len(std_evs[c][0]) / 4) * sample_length
                            fit_max = np.max(pulse_template(t, *par))
                            print('Parameters [t0, An, At, tau_n, tau_in, tau_t]:\n', par)
                            if not np.isclose(fit_max, 0, rtol=3e-3):
                                par[1] /= fit_max
                                par[2] /= fit_max
                    else:
                        par = [0, 0, 0, 1, 1, 1]

                    std_evs[c].append(par)

            sev.require_dataset('event',
                                shape=(self.nmbr_channels, len(std_evs[0][0])),  # this is then length of sev
                                dtype='f')
            sev['event'][...] = np.array([x[0] for x in std_evs])
            sev.require_dataset('fitpar',
                                shape=(self.nmbr_channels, len(std_evs[0][1])),
                                dtype='float')
            sev['fitpar'][...] = np.array([x[1] for x in std_evs])

            # description of the fitparameters (data=column_in_fitpar)
            sev['fitpar'].attrs.create(name='t_0', data=0)
            sev['fitpar'].attrs.create(name='A_n', data=1)
            sev['fitpar'].attrs.create(name='A_t', data=2)
            sev['fitpar'].attrs.create(name='tau_n', data=3)
            sev['fitpar'].attrs.create(name='tau_in', data=4)
            sev['fitpar'].attrs.create(name='tau_t', data=5)

            mp = np.array([calc_main_parameters(x[0]).getArray() for x in std_evs])

            sev.require_dataset('mainpar',
                                shape=mp.shape,
                                dtype='float')

            sev['mainpar'][...] = mp

            # description of the mainpar (data=col_in_mainpar)
            sev['mainpar'].attrs.create(name='pulse_height', data=0)
            sev['mainpar'].attrs.create(name='t_zero', data=1)
            sev['mainpar'].attrs.create(name='t_rise', data=2)
            sev['mainpar'].attrs.create(name='t_max', data=3)
            sev['mainpar'].attrs.create(name='t_decaystart', data=4)
            sev['mainpar'].attrs.create(name='t_half', data=5)
            sev['mainpar'].attrs.create(name='t_end', data=6)
            sev['mainpar'].attrs.create(name='offset', data=7)
            sev['mainpar'].attrs.create(name='linear_drift', data=8)
            sev['mainpar'].attrs.create(name='quadratic_drift', data=9)

            print('{} SEV calculated.'.format(type))

    def calc_of(self, down: int = 1, name_appendix: str = '', window: bool = True, use_this_sev: list = None):
        """
        Calculate the Optimum Filer from the NPS and the SEV.

        The data format and method was described in "(2018) N. Ferreiro Iachellini, Increasing the sensitivity to
        low mass dark matter in cresst-iii witha new daq and signal processing", doi 10.5282/edoc.23762.

        :param down: The downsample factor of the optimal filter transfer function.
        :type down: int
        :param name_appendix: A string that is appended to the group name stdevent and optimumfilter.
        :type name_appendix: string
        :param window: Include a window function to the standard event before building the filter.
        :type window: bool
        :param use_this_sev: Here you can hand an alternativ list of standard events for all channels, in case you
            do not want to use one that is stored in the HDF5 set.
        :type use_this_sev: list
        """

        with h5py.File(self.path_h5, 'r+') as h5f:
            if use_this_sev is None:
                stdevent_pulse = np.array([h5f['stdevent' + name_appendix]['event'][i]
                                           for i in range(self.nmbr_channels)])
            else:
                stdevent_pulse = use_this_sev

            mean_nps = np.array([h5f['noise']['nps'][i] for i in range(self.nmbr_channels)])

            if down > 1:
                stdevent_pulse = np.mean(stdevent_pulse.reshape(-1, int(len(stdevent_pulse[1]) / down), down), axis=2)
                first_nps_val = mean_nps[:, 0]
                mean_nps = mean_nps[:, 1:]
                mean_nps = np.mean(mean_nps.reshape(-1, int(len(mean_nps[1]) / down), down), axis=2)
                mean_nps = np.concatenate((first_nps_val.reshape(-1, 1), mean_nps), axis=1)

            print('CREATE OPTIMUM FILTER.')

            of = np.array([optimal_transfer_function(
                stdevent_pulse[i], mean_nps[i], window) for i in range(self.nmbr_channels)])

            optimumfilter = h5f.require_group('optimumfilter' + name_appendix)
            if down > 1:
                optimumfilter.require_dataset('optimumfilter_real_down{}'.format(down),
                                              dtype='float',
                                              shape=of.real.shape)
                optimumfilter.require_dataset('optimumfilter_imag_down{}'.format(down),
                                              dtype='float',
                                              shape=of.real.shape)

                optimumfilter['optimumfilter_real_down{}'.format(down)][...] = of.real
                optimumfilter['optimumfilter_imag_down{}'.format(down)][...] = of.imag
            else:
                optimumfilter.require_dataset('optimumfilter_real',
                                              dtype='float',
                                              shape=of.real.shape)
                optimumfilter.require_dataset('optimumfilter_imag',
                                              dtype='float',
                                              shape=of.real.shape)

                optimumfilter['optimumfilter_real'][...] = of.real
                optimumfilter['optimumfilter_imag'][...] = of.imag

            print('OF updated.')

    # apply the optimum filter
    def apply_of(self, type='events', name_appendix_group: str = '', name_appendix_set: str = '',
                 chunk_size=10000, hard_restrict=False, down=1, window=True, first_channel_dominant=False,
                 baseline_model='constant', pretrigger_samples=500, onset_to_dominant_channel=None):
        """
        Calculates the height of events or testpulses after applying the optimum filter.

        :param type: The group name in the HDF5 set, either events of testpulses.
        :type type: string
        :param name_appendix_group: A string that is appended to the stdevent group in the HDF5 set.
        :type name_appendix_group: string
        :param name_appendix_set: A string that is appended to the of_ph set in the HDF5 set.
        :type name_appendix_set: string
        :param chunk_size: The size how many events are processes simultaneously to avoid memory error.
        :type chunk_size: int
        :param hard_restrict: If True, the maximum search is restricted to 20-30% of the record window.
        :type hard_restrict: bool
        :param down: The events get downsampled with this factor before application of the filter.
        :type down: int
        :param window: If true, a window function is applied to the record window, before filtering. This is recommended,
            to avoid artifacts from left-right offset differences of the baseline.
        :type window: bool
        :param first_channel_dominant: Take the maximum position from the first channel and evaluate the others at the
            same position.
        :type first_channel_dominant: bool
        :param baseline_model: Either 'constant', 'linear' or 'exponential'. The baseline model substracted from all
            events.
        :type baseline_model: str
        :param pretrigger_samples: The number of samples from start of the record window that are considered the pre
            trigger region.
        :type pretrigger_samples: int
        :param onset_to_dominant_channel: The difference in the onset value to the dominant channel. If e.g. the second
            channel has a typical max_pos value of 4000, but the first of 4100, then the onset for this would be -100.
        :type onset_to_dominant_channel: list of ints
        """

        print('Calculating OF Heights.')

        if onset_to_dominant_channel is None:
            onset_to_dominant_channel = np.zeros(self.nmbr_channels)
        assert len(onset_to_dominant_channel) == self.nmbr_channels, \
            'onset_to_dominant_channel must have length nmbr_channels!'

        with h5py.File(self.path_h5, 'r+') as f:
            events = f[type]['event']
            sev = np.array(f['stdevent' + name_appendix_group]['event'])
            nps = np.array(f['noise']['nps'])
            if 'optimumfilter' + name_appendix_group in f:
                transfer_function = np.array(f['optimumfilter' + name_appendix_group]['optimumfilter_real']) + \
                                    1j * np.array(f['optimumfilter' + name_appendix_group]['optimumfilter_imag'])
            else:
                transfer_function = None

            if 'of_ph' + name_appendix_set in f[type]:
                del f[type]['of_ph' + name_appendix_set]

            f[type].require_dataset(name='of_ph' + name_appendix_set,
                                    shape=(self.nmbr_channels, len(events[0])),
                                    dtype='float')

            nmbr_events = len(events[0])
            counter = 0

            # we do the calculation in batches, so that memory does not overflow
            while counter + chunk_size < nmbr_events:
                for c in range(self.nmbr_channels):
                    if first_channel_dominant and c == 0:
                        of_ph, peakpos = get_amplitudes(events[c, counter:counter + chunk_size], sev[c], nps[c],
                                                        hard_restrict=hard_restrict, down=down, window=window,
                                                        return_peakpos=True,
                                                        baseline_model=baseline_model,
                                                        pretrigger_samples=pretrigger_samples,
                                                        transfer_function=transfer_function[c])
                    elif first_channel_dominant:
                        of_ph = get_amplitudes(events[c, counter:counter + chunk_size], sev[c], nps[c],
                                               hard_restrict=hard_restrict, down=down, window=window,
                                               peakpos=peakpos + onset_to_dominant_channel[c],
                                               return_peakpos=False,
                                               baseline_model=baseline_model, pretrigger_samples=pretrigger_samples,
                                               transfer_function=transfer_function[c])
                    else:
                        of_ph = get_amplitudes(events[c, counter:counter + chunk_size], sev[c], nps[c],
                                               hard_restrict=hard_restrict, down=down, window=window,
                                               return_peakpos=False,
                                               baseline_model=baseline_model, pretrigger_samples=pretrigger_samples,
                                               transfer_function=transfer_function[c])

                    f[type]['of_ph' + name_appendix_set][c, counter:counter + chunk_size] = of_ph
                counter += chunk_size

            # calc rest that is smaller than a batch
            for c in range(self.nmbr_channels):
                if first_channel_dominant and c == 0:
                    of_ph, peakpos = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c],
                                                    hard_restrict=hard_restrict, down=down, window=window,
                                                    return_peakpos=True,
                                                    baseline_model=baseline_model,
                                                    pretrigger_samples=pretrigger_samples,
                                                    transfer_function=transfer_function[c])
                elif first_channel_dominant:
                    of_ph = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c],
                                           hard_restrict=hard_restrict, down=down, window=window,
                                           peakpos=peakpos + onset_to_dominant_channel[c], return_peakpos=False,
                                           baseline_model=baseline_model, pretrigger_samples=pretrigger_samples,
                                           transfer_function=transfer_function[c])
                else:
                    of_ph = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c],
                                           hard_restrict=hard_restrict, down=down, window=window,
                                           return_peakpos=False,
                                           baseline_model=baseline_model, pretrigger_samples=pretrigger_samples,
                                           transfer_function=transfer_function[c])
                f[type]['of_ph' + name_appendix_set][c, counter:nmbr_events] = of_ph

    # calc stdevent carrier
    def calc_exceptional_sev(self,
                             naming,
                             channel=0,
                             type='events',
                             use_prediction_instead_label=False,
                             model=None,
                             correct_label=None,
                             use_idx=None,
                             pulse_height_interval=[0, 10],
                             left_right_cutoff=None,
                             rise_time_interval=None,
                             decay_time_interval=None,
                             onset_interval=None,
                             remove_offset=True,
                             verb=True,
                             scale_fit_height=True,
                             sample_length=None):
        """
        Calculate an exceptional Standard Event for a Class in the HDF5 File, for only one specific channel.

        Attention! Since v1.0 can as well use the regular calc_sev method for calculating SEVs of different pulse shapes.
        This method is therefore no longer maintained.

        :param naming: Pick a name for the type of event, e.g. 'carrier'.
        :type naming: string
        :param channel: The number of the channel in the hdf5 file.
        :type channel: int
        :param type: The group name in the HDF5 set, either "events" or "testpulses".
        :type type: string
        :param use_prediction_instead_label: If True then instead of the labels the predictions are used.
        :type use_prediction_instead_label: bool
        :param model: If set this is the name of the model whiches predictions are in the
            h5 file, e.g. "RF" --> look for "RF_predictions".
        :type model: string or None
        :param correct_label: Use only events with this label.
        :type correct_label: int or None
        :param use_idx: If set then only these indices are used for the sev creation.
        :type use_idx: list of ints or None
        :param pulse_height_interval: The upper
            and lower bound for the pulse heights to include into the creation of the SEV.
        :type pulse_height_interval: list of length 2 (interval)
        :param left_right_cutoff: The maximal abs value of the linear slope of events
            to be included in the Sev calculation. Based on the sample index as x-values.
        :type left_right_cutoff: float
        :param rise_time_interval: The upper
            and lower bound for the rise time to include into the creation of the SEV.
            Based on the sample index as x-values.
        :type rise_time_interval: lists of length 2 (interval)
        :param decay_time_interval: The upper
            and lower bound for the decay time to include into the creation of the SEV.
            Based on the sample index as x-values.
        :type decay_time_interval: list of length 2 (interval)
        :param onset_interval: The upper
            and lower bound for the onset time to include into the creation of the SEV.
            Based on the sample index as x-values.
        :type onset_interval: list of length 2 (interval)
        :param remove_offset: If True the offset is removed before the events are superposed for the
            sev calculation. Highly recommended!
        :type remove_offset: bool
        :param verb: If True, some verbal feedback is output about the progress of the method.
        :type verb: bool
        :param scale_fit_height: If True the parametric fit to the sev is normalized to height 1 after
            the fit is done.
        :type scale_fit_height: bool
        :param sample_length: The length of one sample in milliseconds. If None, this is calculated from the sample
            frequency.
        :type sample_length: float
        """

        warn(
            'Attention! Since v1.0 can as well use the regular calc_sev method for calculating SEVs of different pulse shapes. This method is therefore no longer maintained.')

        if sample_length is None:
            sample_length = 1000 / self.sample_frequency

        with h5py.File(self.path_h5, 'r+') as h5f:

            if correct_label is None and use_idx is None:
                raise KeyError('Provide either Correct Label or Index List!')

            if correct_label is not None:
                if use_prediction_instead_label:
                    if model is not None:
                        labels = h5f[type]['{}_predictions'.format(model)][channel]
                    else:
                        raise KeyError('Please provide a model string!')
                else:
                    labels = h5f[type]['labels'][channel]
            else:
                labels = None

            if use_idx is None:
                use_idx = [i for i in range(len(labels))]

            events = h5f[type]['event'][channel, use_idx, :]
            mainpar = h5f[type]['mainpar'][channel, use_idx, :]
            if labels is not None:
                labels = labels[use_idx]

            sev = h5f.require_group('stdevent_{}'.format(naming))

            sev_event, par = generate_standard_event(events=events,
                                                     main_parameters=mainpar,
                                                     labels=labels,
                                                     correct_label=correct_label,
                                                     pulse_height_interval=pulse_height_interval,
                                                     left_right_cutoff=left_right_cutoff,
                                                     rise_time_interval=rise_time_interval,
                                                     decay_time_interval=decay_time_interval,
                                                     onset_interval=onset_interval,
                                                     remove_offset=remove_offset,
                                                     verb=verb,
                                                     scale_fit_height=scale_fit_height,
                                                     sample_length=sample_length)

            sev.require_dataset('event',
                                shape=(len(sev_event),),  # this is then length of sev
                                dtype='f')
            sev['event'][...] = sev_event
            sev.require_dataset('fitpar',
                                shape=(len(par),),
                                dtype='float')
            sev['fitpar'][...] = par

            # description of the fitparameters (data=column_in_fitpar)
            sev['fitpar'].attrs.create(name='t_0', data=0)
            sev['fitpar'].attrs.create(name='A_n', data=1)
            sev['fitpar'].attrs.create(name='A_t', data=2)
            sev['fitpar'].attrs.create(name='tau_n', data=3)
            sev['fitpar'].attrs.create(name='tau_in', data=4)
            sev['fitpar'].attrs.create(name='tau_t', data=5)

            mp = calc_main_parameters(sev_event).getArray()

            sev.require_dataset('mainpar',
                                shape=mp.shape,
                                dtype='f')

            sev['mainpar'][...] = mp

            # description of the mainpar (data=col_in_mainpar)
            sev['mainpar'].attrs.create(name='pulse_height', data=0)
            sev['mainpar'].attrs.create(name='t_zero', data=1)
            sev['mainpar'].attrs.create(name='t_rise', data=2)
            sev['mainpar'].attrs.create(name='t_max', data=3)
            sev['mainpar'].attrs.create(name='t_decaystart', data=4)
            sev['mainpar'].attrs.create(name='t_half', data=5)
            sev['mainpar'].attrs.create(name='t_end', data=6)
            sev['mainpar'].attrs.create(name='offset', data=7)
            sev['mainpar'].attrs.create(name='linear_drift', data=8)
            sev['mainpar'].attrs.create(name='quadratic_drift', data=9)

            print('{} SEV calculated.'.format(type))

    def calc_nps(self, use_labels=False, down=1, percentile=None,
                 rms_cutoff=None, cut_flag=None, window=True, force_zero=True):
        """
        Calculates the mean Noise Power Spectrum with option to use only the baselines
        that are labeled as noise (label == 3).

        :param use_labels: If True only baselines that are labeled as noise are included.
        :type use_labels: bool
        :param down: A factor by that the baselines are downsampled before the calculation - must be 2^x.
        :type down: int
        :param percentile: The lower percentile of the fit errors of the baselines that we include in the calculation.
        :type percentile: int
        :param rms_cutoff: Only baselines with a fit rms below this values are included in the NPS calculation. This
            will overwrite the percentile argument, if it is not set to None.
        :type rms_cutoff: list of nmbr_channels floats
        :param cut_flag: Only the noise baselines for which the value in this array is True, are used for the
            calculation.
        :type cut_flag: 1d bool array
        :param window: If True, a window function is applied to the noise baselines before the calculation of the NPS.
        :type window: bool
        :param force_zero: Force the zero coefficient (constant offset) of the NPS to zero.
        :type force_zero: bool
        """
        print('Calculate NPS.')

        if rms_cutoff is None:
            rms_cutoff = [None for c in range(self.nmbr_channels)]

        # open file
        with h5py.File(self.path_h5, 'r+') as h5f:
            baselines = np.array(h5f['noise']['event'])

            mean_nps = []
            for c in range(self.nmbr_channels):
                bl = baselines[c]
                if use_labels and cut_flag is None:
                    labels = np.array(h5f['noise']['labels'][c])
                    bl = bl[labels[c] == 3]
                elif use_labels and cut_flag is not None:
                    labels = np.array(h5f['noise']['labels'][c])
                    labels = labels[cut_flag]
                    bl = bl[cut_flag]
                    bl = bl[labels[c] == 3]
                elif not use_labels and cut_flag is None:
                    pass
                elif not use_labels and cut_flag is not None:
                    bl = bl[cut_flag]

                if 'fit_rms' in h5f['noise']:
                    rms_baselines = h5f['noise']['fit_rms'][c]
                    if cut_flag is not None:
                        rms_baselines = rms_baselines[cut_flag]

                else:
                    rms_baselines = None
                mean_nps.append(calculate_mean_nps(bl,
                                                   down=down,
                                                   percentile=percentile,
                                                   rms_baselines=rms_baselines,
                                                   sample_length=1 / self.sample_frequency,
                                                   rms_cutoff=rms_cutoff[c],
                                                   window=window)[0])

            mean_nps = np.array([mean_nps[i] for i in range(self.nmbr_channels)])
            frequencies = np.fft.rfftfreq(self.record_length, d=1. / self.sample_frequency * down)

            if force_zero:
                mean_nps[:, 0] = 0

            naming = 'nps'
            naming_fq = 'freq'
            if down > 1:
                naming += '_down' + str(down)
                naming_fq += '_down' + str(down)

            h5f['noise'].require_dataset(naming,
                                         shape=mean_nps.shape,
                                         dtype='float')
            h5f['noise'][naming][...] = mean_nps
            h5f['noise'].require_dataset(naming_fq,
                                         shape=frequencies.shape,
                                         dtype='float')
            h5f['noise'][naming_fq][...] = frequencies

    def calc_additional_mp(self, type='events', path_h5=None, down=1, no_of=False):
        """
        Calculate the additional Main Parameters for the Events in an HDF5 File.

        :param type: The group name within the HDF5 file, either events or testpulses.
        :type type: string
        :param path_h5: An alternative full path to the hdf5 file, e.g. "data/bck_001.h5".
        :type path_h5: string
        :param down: The downsample rate before calculating the parameters.
        :type down: int
        :param no_of: Do not use the optimum filter, fill the quantities with zeros instead.
        :type no_of: bool
        """

        if not path_h5:
            path_h5 = self.path_h5

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]

            assert 'optimumfilter' in h5f or no_of, 'You need to calculate the optimal filter first, or activate no_of!'

            if not no_of:
                of_real = np.array(h5f['optimumfilter']['optimumfilter_real'])
                of_imag = np.array(h5f['optimumfilter']['optimumfilter_imag'])
                of = of_real + 1j * of_imag
            else:
                of = [None for i in range(self.nmbr_channels)]

            print('CALCULATE ADDITIONAL MAIN PARAMETERS.')

            add_par_event = []
            for c in range(self.nmbr_channels):
                add_par_event.append([calc_additional_parameters(ev, of[c], down=down) for ev in events['event'][c]])

            add_par_event = np.array(add_par_event)

            events.require_dataset(name='add_mainpar',
                                   shape=(add_par_event.shape),
                                   dtype='float')
            events['add_mainpar'][...] = add_par_event

            events['add_mainpar'].attrs.create(name='array_max', data=0)
            events['add_mainpar'].attrs.create(name='array_min', data=1)
            events['add_mainpar'].attrs.create(name='var_first_eight', data=2)
            events['add_mainpar'].attrs.create(name='mean_first_eight', data=3)
            events['add_mainpar'].attrs.create(name='var_last_eight', data=4)
            events['add_mainpar'].attrs.create(name='mean_last_eight', data=5)
            events['add_mainpar'].attrs.create(name='var', data=6)
            events['add_mainpar'].attrs.create(name='mean', data=7)
            events['add_mainpar'].attrs.create(name='skewness', data=8)
            events['add_mainpar'].attrs.create(name='max_derivative', data=9)
            events['add_mainpar'].attrs.create(name='ind_max_derivative', data=10)
            events['add_mainpar'].attrs.create(name='min_derivative', data=11)
            events['add_mainpar'].attrs.create(name='ind_min_derivative', data=12)
            events['add_mainpar'].attrs.create(name='max_filtered', data=13)
            events['add_mainpar'].attrs.create(name='ind_max_filtered', data=14)
            events['add_mainpar'].attrs.create(name='skewness_filtered_peak', data=15)

    def apply_logical_cut(self,
                          cut_flag: list,
                          naming: str,
                          channel: int,
                          type: str = 'events',
                          delete_old: bool = False):
        """
        Save the cut flag of a logical cut within the HDF5 file.

        :param cut_flag: The cut flag that we want to save.
        :type cut_flag: list of bools
        :param naming: The naming of the dataset to save.
        :type naming: string
        :param channel: The channel for that the cut flag is meant.
        :type channel: int
        :param type: The naming of the group in the HDF5 file, in that we want to save the cut flag, e.g. 'events'.
        :type type: string
        :param delete_old: If true, the old dataset of this name in the group 'type' gets deleted.
        :type delete_old: bool
        """

        with h5py.File(self.path_h5, 'r+') as f:

            if delete_old:
                if naming in f[type]:
                    print('Delete old {} dataset'.format(naming))
                    del f[type][naming]

            cut_dataset = f[type].require_dataset(name=naming,
                                                  shape=(self.nmbr_channels, len(cut_flag)),
                                                  dtype=bool)
            cut_dataset[channel, ...] = cut_flag

        print('Applied logical cut.')

    def include_values(self,
                       values: list,
                       naming: str,
                       channel: int,
                       type: str = 'events',
                       delete_old: bool = False):
        """
        Include values as a data set in the HDF5 file.

        Typically this is used to store values of cuts or calibrated energies.

        :param values: The values that we want to include in the file.
        :type values: list of floats
        :param naming: The name of the data set in the HDF5 file.
        :type naming: string
        :param channel: The channel number to which we want to include the cut values.
        :type channel: int
        :param type: The group name in the HDF5 set.
        :type type: string
        :param delete_old: If a set by this name exists already, it gets deleted first.
        :type delete_old: bool
        """

        with h5py.File(self.path_h5, 'r+') as f:

            if delete_old:
                if naming in f[type]:
                    print('Delete old {} dataset'.format(naming))
                    del f[type][naming]

            cut_dataset = f[type].require_dataset(name=naming,
                                                  shape=(self.nmbr_channels, len(values)),
                                                  dtype=float)
            cut_dataset[channel, ...] = values

        print('Included values.')

    def calc_ph_correlated(self, type='events', dominant_channel=0,
                           offset_to_dominant_channel=None,
                           max_search_range=50,
                           ):
        """
        Calculate the correlated pulse heights of the channels.

        :param events: The events of all channels.
        :type events: 2D array of shape (nmbr_channels, record_length)
        :param dominant_channel: Which channel is the one for the primary max search.
        :type dominant_channel: int
        :param offset_to_dominant_channel: The expected offsets of the peaks of pulses to the pesk of the dominant channel.
        :type offset_to_dominant_channel: list of ints
        :param max_search_range: The number of samples that are included in the search range of the maximum search in the
            non-dominant channels.
        :type max_search_range: int

        >>> import cait as ai

        >>> path_data = '../CRESST_DATA/run36/run36_Gode1/'
        >>> fname = 'stream_bck_003'

        >>> dh_stream = ai.DataHandler(channels=[9, 10, 11, ])
        >>> dh_stream.set_filepath(path_h5=path_data, fname=fname, appendix=False)

        >>> dh_stream.calc_ph_correlated()
        """

        with h5py.File(self.path_h5, 'r+') as f:
            print('CALCULATE CORRELATED PULSE HEIGHTS.')

            nmbr_events = f[type]['event'].shape[1]

            ph_corr = np.empty((self.nmbr_channels, nmbr_events), dtype=float)

            for ev in trange(nmbr_events):
                ph_corr[:, ev] = calc_correlated_ph(f[type]['event'][:, ev],
                                                    dominant_channel=dominant_channel,
                                                    offset_to_dominant_channel=offset_to_dominant_channel,
                                                    max_search_range=max_search_range,
                                                    )

            set_ph_corr = f[type].require_dataset(name='ph_corr',
                                                  shape=ph_corr.shape,
                                                  dtype=float)

            set_ph_corr[...] = ph_corr

    def calc_peakdet(self, type='events', lag=1024, threshold=5, look_ahead=1024):
        """
        Calculate the number of prominent peaks within the record window. A number > 1 points towards pile up events.

        Based on https://stackoverflow.com/a/22640362/15216821.

        :param type: The group name of the HDF5 set.
        :type type: str
        :param lag: The lag value of the algorithm, i.e. the number of samples that are taken to calculate the
            moving mean and standard deviation.
        :type lag: int
        :param threshold:
        :type threshold: int
        :param look_ahead: When a sample triggers, we look for even higher samples in the subsequent look_ahead number
            of samples.
        :type look_ahead: int
        """

        with h5py.File(self.path_h5, 'r+') as f:

            print('CALCULATE NUMBER OF PEAKS.')
            nmbr_events = f[type]['event'].shape[1]
            nmbr_peaks = np.empty((self.nmbr_channels, nmbr_events), dtype=float)

            for c in range(self.nmbr_channels):
                for i in trange(nmbr_events):
                    signal, _, _, _ = get_triggers(array=f[type]['event'][c, i],
                                                   lag=lag,
                                                   threshold=threshold,
                                                   init_mean=None,
                                                   init_var=None,
                                                   look_ahead=look_ahead)
                    nmbr_peaks[c, i] = len(signal)

            set_peaks = f[type].require_dataset(name='nmbr_peaks',
                                                shape=nmbr_peaks.shape,
                                                dtype=float)

            set_peaks[...] = nmbr_peaks
