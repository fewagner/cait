# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from multiprocessing import Pool
from ..features._mp import calc_main_parameters, calc_additional_parameters
from ..filter._of import optimal_transfer_function
from ..fit._sev import generate_standard_event
from ..filter._of import get_amplitudes

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
    def calc_mp(self, type='events', path_h5=None, processes=4, down=1, max_bounds=None):
        """
        Calculate the Main Parameters for the Events in an HDF5 File.

        :param type: The group in the HDF5 set, either events or testpulses.
        :param type: string
        :param path_h5: An alternative full path to a hdf5 file, e.g. "data/bck_001.h5".
        :param path_h5: string or None
        :param processes: The number of processes to use for the calculation.
        :param processes: int
        :param down: The events get downsampled by this factor for the calculation of main parameters.
        :param down: int
        :param: max_bounds: The interval of indices to which we restrict the maximum search for the pulse height.
        :param: max_bounds: tuple of two ints
        """

        if not path_h5:
            path_h5 = self.path_h5

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]
            nmbr_ev = len(events['event'][0])

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
                 verb=True,
                 scale_fit_height=True,
                 sample_length=None,
                 t0_start=None,
                 opt_start=False):
        """
        Calculate the Standard Event for the Events in the HDF5 File.

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
        :param left_right_cutoff: The maximal abs value of the linear slope of events
            to be included in the Sev calculation. The slope is calculated with respect to the sample index as x-values.
        :type left_right_cutoff: list of NMBR_CHANNELS floats
        :param rise_time_interval: The upper
            and lower bound for the rise time to include into the creation of the SEV.
            based on the sample index as x-values.
        :type rise_time_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param decay_time_interval: The upper
            and lower bound for the decay time to include into the creation of the SEV.
            Based on the sample index as x-values.
        :type decay_time_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param onset_interval:  The upper
            and lower bound for the onset time to include into the creation of the SEV.
            Based on the sample index as x-values.
        :type onset_interval: list of NMBR_CHANNELS lists of length 2 (intervals)
        :param remove_offset: Tf True the offset is removed before the events are superposed for the
            sev calculation. Highly recommended!
        :type remove_offset: bool
        :param verb: If True some verbal feedback is output about the progress of the method.
        :type verb: bool
        :param scale_fit_height: If True the parametric fit to the sev is normalized to height 1 after
            the fit is done.
        :type scale_fit_height: bool
        :param sample_length: The length of one sample in milliseconds. If None, this is calculated from the sample
            frequency.
        :type sample_length: float
        :param t0_start: The start values for t0 in the fit.
        :type t0_start: 2-tupel of floats
        :param opt_start: If true, a pre-fit is applied to find optimal start values.
        :type opt_start: bool
        """

        if sample_length is None:
            sample_length = 1/self.sample_frequency * 1000

        with h5py.File(self.path_h5, 'r+') as h5f:
            events = h5f[type]['event']
            mainpar = h5f[type]['mainpar']

            std_evs = []

            # set the start values for t0
            if t0_start is None:
                t0_start = [None for i in range(self.nmbr_channels)]

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
                use_idx = list(range(len(events[0])))

            for c in range(self.nmbr_channels):
                print('')
                print('Calculating SEV for Channel {}'.format(c))
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
                                                       sample_length=sample_length,
                                                       t0_start=t0_start[c],
                                                       opt_start=opt_start,
                                                       ))

            sev.require_dataset('event',
                                shape=(self.nmbr_channels, len(std_evs[0][0])),  # this is then length of sev
                                dtype='f')
            sev['event'][...] = np.array([x[0] for x in std_evs])
            sev.require_dataset('fitpar',
                                shape=(self.nmbr_channels, len(std_evs[0][1])),
                                dtype='f')
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

    def calc_of(self, down: int = 1, name_appendix: str = '', window=True):
        """
        Calculate the Optimum Filer from the NPS and the SEV.

        :param down: The downsample factor of the optimal filter transfer function.
        :type down: int
        :param name_appendix: A string that is appended to the group name stdevent and optimumfilter.
        :type name_appendix: string
        :param window: Include a window function to the standard event before building the filter.
        :type window: bool
        """

        with h5py.File(self.path_h5, 'r+') as h5f:
            stdevent_pulse = np.array([h5f['stdevent' + name_appendix]['event'][i]
                                       for i in range(self.nmbr_channels)])
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
                                              dtype='f',
                                              shape=of.real.shape)
                optimumfilter.require_dataset('optimumfilter_imag_down{}'.format(down),
                                              dtype='f',
                                              shape=of.real.shape)

                optimumfilter['optimumfilter_real_down{}'.format(down)][...] = of.real
                optimumfilter['optimumfilter_imag_down{}'.format(down)][...] = of.imag
            else:
                optimumfilter.require_dataset('optimumfilter_real',
                                              dtype='f',
                                              shape=of.real.shape)
                optimumfilter.require_dataset('optimumfilter_imag',
                                              dtype='f',
                                              shape=of.real.shape)

                optimumfilter['optimumfilter_real'][...] = of.real
                optimumfilter['optimumfilter_imag'][...] = of.imag

            print('OF updated.')

    # apply the optimum filter
    def apply_of(self, type='events', name_appendix_group: str = '', name_appendix_set: str = '',
                 chunk_size=10000, hard_restrict=False, down=1, window=True, first_channel_dominant=False):
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
        """

        print('Calculating OF Heights.')

        with h5py.File(self.path_h5, 'r+') as f:
            events = f[type]['event']
            sev = np.array(f['stdevent' + name_appendix_group]['event'])
            nps = np.array(f['noise']['nps'])

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
                                               return_peakpos=True)
                    elif first_channel_dominant:
                        of_ph = get_amplitudes(events[c, counter:counter + chunk_size], sev[c], nps[c],
                                                         hard_restrict=hard_restrict, down=down, window=window,
                                                         peakpos=peakpos)
                    else:
                        of_ph = get_amplitudes(events[c, counter:counter + chunk_size], sev[c], nps[c],
                                               hard_restrict=hard_restrict, down=down, window=window)

                    f[type]['of_ph' + name_appendix_set][c, counter:counter + chunk_size] = of_ph
                counter += chunk_size

            # calc rest that is smaller than a batch
            for c in range(self.nmbr_channels):
                if first_channel_dominant and c == 0:
                    of_ph, peakpos = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c],
                                                    hard_restrict=hard_restrict, down=down, window=window,
                                                    return_peakpos=True)
                elif first_channel_dominant:
                    of_ph = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c],
                                           hard_restrict=hard_restrict, down=down, window=window,
                                           peakpos=peakpos, return_peakpos=False)
                else:
                    of_ph = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c],
                                           hard_restrict=hard_restrict, down=down, window=window,
                                           return_peakpos=False)
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
                                dtype='f')
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

    def calc_nps(self, use_labels=False, down=1, percentile=50):
        """
        Calculates the mean Noise Power Spectrum with option to use only the baselines
        that are labeled as noise (label == 3).

        :param use_labels: If True only baselines that are labeled as noise are included.
        :type use_labels: bool
        :param down: A factor by that the baselines are downsampled before the calculation - must be 2^x.
        :type down: int
        :param percentile: The lower percentile of the fit errors of the baselines that we include in the calculation.
        :type percentile: int
        """
        print('Calculate NPS.')

        # open file
        with h5py.File(self.path_h5, 'r+') as h5f:
            baselines = np.array(h5f['noise']['event'])
            if use_labels:
                labels = np.array(h5f['noise']['labels'])

            mean_nps = []
            for c in range(self.nmbr_channels):
                bl = baselines[c]
                if use_labels:
                    bl = bl[labels[c] == 3]  # 3 is noise label
                if 'fit_rms' in h5f['noise']:
                    rms_baselines = h5f['noise']['fit_rms'][c]
                else:
                    rms_baselines = None
                mean_nps.append(calculate_mean_nps(bl,
                                                   down=down,
                                                   percentile=percentile,
                                                   rms_baselines=rms_baselines)[0])

            mean_nps = np.array([mean_nps[i] for i in range(self.nmbr_channels)])
            frequencies = np.fft.rfftfreq(self.record_length, d=1. / self.sample_frequency * down)

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

    def calc_additional_mp(self, type='events', path_h5=None, down=1):
        """
        Calculate the additional Main Parameters for the Events in an HDF5 File.

        :param type: The group name within the HDF5 file, either events or testpulses.
        :type type: string
        :param path_h5: An alternative full path to the hdf5 file, e.g. "data/bck_001.h5".
        :type path_h5: string
        :param down: The downsample rate before calculating the parameters.
        :type down: int
        """

        if not path_h5:
            path_h5 = self.path_h5

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]

            of_real = np.array(h5f['optimumfilter']['optimumfilter_real'])
            of_imag = np.array(h5f['optimumfilter']['optimumfilter_imag'])
            of = of_real + 1j * of_imag

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
