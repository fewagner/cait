# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
from multiprocessing import Pool
from ..features._mp import calc_main_parameters, calc_additional_parameters
from ..filter._of import optimal_transfer_function
from ..fit._sev import generate_standard_event
from ..features._ts_feat import calc_ts_features
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

    # Recalculate MP
    def recalc_mp(self, type, path_h5=None, processes=4):
        """
        Calculate the Main Parameters for the Events in an HDF5 File.

        :param type: string, either events or testpulses
        :param path_h5: string, optional, the full path to the hdf5 file, e.g. "data/bck_001.h5"
        :param processes: int, the number of processes to use for the calculation
        :return: -
        """

        if not path_h5:
            path_h5 = self.path_h5

        h5f = h5py.File(path_h5, 'r+')
        events = h5f[type]

        print('CALCULATE MAIN PARAMETERS.')

        with Pool(processes) as p:  # basically a for loop running on 4 processes
            mainpar_list_event = []
            for c in range(self.nmbr_channels):
                mainpar_list_event.append(p.map(
                    calc_main_parameters, events['event'][c, :, :]))
        mainpar_event = np.array([[o.getArray() for o in element] for element in mainpar_list_event])

        events.require_dataset(name='mainpar',
                               shape=(mainpar_event.shape),
                               dtype='float')
        events['mainpar'][...] = mainpar_event


    # calc stdevent testpulses
    def recalc_sev(self,
                   type='events',
                   use_labels=True,
                   correct_label=None,
                   pulse_height_intervall=[[0.5, 1.5], [0.5, 1.5]],
                   left_right_cutoff=None,
                   rise_time_intervall=None,
                   decay_time_intervall=None,
                   onset_intervall=None,
                   remove_offset=True,
                   verb=True,
                   scale_fit_height=True,
                   sample_length=0.04):
        """
        Calculate the Standard Event for the Events in the HDF5 File.

        :param type: string, either "events" or "testpulses"
        :param use_labels: bool, if True a labels file must be included in the hdf5 file,
            then only the events labeled as events or testpulses are included in the calculation
        :param pulse_height_intervall: list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the pulse heights to include into the creation of the SEV
        :param left_right_cutoff: list of NMBR_CHANNELS floats, the maximal abs value of the linear slope of events
            to be included in the Sev calculation; based on the sample index as x-values
        :param rise_time_intervall: list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the rise time to include into the creation of the SEV;
            based on the sample index as x-values
        :param decay_time_intervall: list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the decay time to include into the creation of the SEV;
            based on the sample index as x-values
        :param onset_intervall:  list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the onset time to include into the creation of the SEV;
            based on the sample index as x-values
        :param remove_offset: bool, if True the offset is removed before the events are superposed for the
            sev calculation; highly recommended!
        :param verb: bool, if True some verbal feedback is output about the progress of the method
        :param scale_fit_height: bool, if True the parametric fit to the sev is normalized to height 1 after
            the fit is done
        :param sample_length: float, the length of one sample in milliseconds
        :return: -
        """

        h5f = h5py.File(self.path_h5, 'r+')
        events = h5f[type]['event']
        mainpar = h5f[type]['mainpar']

        std_evs = []

        # fix the issue with different arguments for different channels
        inp = [left_right_cutoff, rise_time_intervall, decay_time_intervall, onset_intervall]
        for i, var in enumerate(inp):
            if var is None:
                inp[i] = [None for c in range(self.nmbr_channels)]

        if use_labels:
            labels = h5f[type]['labels']
        else:
            labels = [None for c in range(self.nmbr_channels)]

        if correct_label is None:
            if type == 'events':
                sev = h5f.require_group('stdevent')
                correct_label = 1
            elif type == 'testpulses':
                sev = h5f.require_group('stdevent_tp')
                correct_label = 2
            else:
                raise NotImplementedError('Type must be events or testpulses!')
        else:
            sev = h5f.require_group('stdevent_{}'.format(correct_label))

        for c in range(self.nmbr_channels):
            std_evs.append(generate_standard_event(events=events[c, :, :],
                                                   main_parameters=mainpar[c, :, :],
                                                   labels=labels[c],
                                                   correct_label=correct_label,
                                                   pulse_height_intervall=pulse_height_intervall[c],
                                                   left_right_cutoff=inp[0][c],
                                                   rise_time_intervall=inp[1][c],
                                                   decay_time_intervall=inp[2][c],
                                                   onset_intervall=inp[3][c],
                                                   remove_offset=remove_offset,
                                                   verb=verb,
                                                   scale_fit_height=scale_fit_height,
                                                   sample_length=sample_length))

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

        h5f.close()

    def recalc_of(self):
        """
        Calculate the Optimum Filer from the NPS and the SEV

        :return: -
        """

        h5f = h5py.File(self.path_h5, 'r+')
        p_stdevent_pulse = h5f['stdevent']['event'][0]
        p_mean_nps = h5f['noise']['nps'][0]
        l_stdevent_pulse = h5f['stdevent']['event'][1]
        l_mean_nps = h5f['noise']['nps'][1]

        print('CREATE OPTIMUM FILTER.')

        of = np.array([optimal_transfer_function(p_stdevent_pulse, p_mean_nps),
                       optimal_transfer_function(l_stdevent_pulse, l_mean_nps)])

        optimumfilter = h5f.require_group('optimumfilter')
        optimumfilter.require_dataset('optimumfilter_real',
                                      dtype='f',
                                      shape=of.real.shape)
        optimumfilter.require_dataset('optimumfilter_imag',
                                      dtype='f',
                                      shape=of.real.shape)

        optimumfilter['optimumfilter_real'][...] = of.real
        optimumfilter['optimumfilter_imag'][...] = of.imag

        print('OF updated.')

        h5f.close()

    # calculate TS Features
    def calc_features(self, type='events', downsample=None):
        """
        Calcuate the TimeSeries features from the library tsfel and store in the HDF5 dataset

        :param type: string, either events, testpulses or noise
        :param downsample: int, the factor to downsample the events before the calculation;
            in experiments, a downsampling to a total event length of 256 provided the same
            results as working with the high sample frequency, but saved large amounts of runtime
        :return: -
        """

        f = h5py.File(self.path_h5, 'r+')
        events = np.array(f[type]['event'])
        features = []

        if downsample is None:
            downsample = self.down

        for c in range(self.nmbr_channels):
            features.append(calc_ts_features(events=events[c],
                                             nmbr_channels=self.nmbr_channels,
                                             nmbr_events=len(events[0]),
                                             record_length=self.record_length,
                                             down=downsample,
                                             sample_frequency=self.sample_frequency,
                                             scaler=None))

        features = np.array(features)

        print('Features calculated.')

        f[type].require_dataset('ts_features',
                                shape=features.shape,
                                dtype='f')

        f[type]['ts_features'][...] = features

        f.close()

    # apply the optimum filter
    def apply_of(self, type='events'):
        """
        Calculates the height of events or testpulses after applying the optimum filter

        :param type: string, either events of testpulses
        :return: -
        """

        print('Calculating OF Heights.')

        f = h5py.File(self.path_h5, 'r+')
        events = np.array(f[type]['event'])
        sev = np.array(f['stdevent']['event'])
        nps = np.array(f['noise']['nps'])
        of = np.zeros((self.nmbr_channels, int(self.record_length / 2 + 1)), dtype=complex)
        of.real = f['optimumfilter']['optimumfilter_real']
        of.imag = f['optimumfilter']['optimumfilter_imag']

        of_ph = []
        for c in range(self.nmbr_channels):
            of_ph.append(get_amplitudes(events[c], sev[c], nps[c]))

        of_ph = np.array([c for c in of_ph])

        f[type].require_dataset(name='of_ph',
                                shape=(self.nmbr_channels, len(events[0])),
                                dtype='float')
        f[type]['of_ph'][...] = of_ph

        f.close()


    # calc stdevent carrier
    def calc_exceptional_sev(self,
                             naming,
                             channel=0,
                             type='events',
                             use_prediction_instead_label=False,
                             model=None,
                             correct_label=None,
                             idx_list=None,
                             pulse_height_intervall=[0.5, 1.5],
                             left_right_cutoff=None,
                             rise_time_intervall=None,
                             decay_time_intervall=None,
                             onset_intervall=None,
                             remove_offset=True,
                             verb=True,
                             scale_fit_height=True,
                             sample_length=0.04):
        """
        Calculate an exceptional Standard Event for a Class in the HDF5 File, for only one specific channel.

        :param naming: string, pick a name for the type of event
        :param channel: int, the number of the channel in the hdf5 file
        :param type: string, either "events" or "testpulses"
        :param use_prediction_instead_label: bool, if True then instead of the labels the predictions are used
        :param model: string or None, if set this is the name of the model whiches predictions are in the
            h5 file, e.g. "RF" --> look for "RF_predictions"
        :param correct_label: int or None, if not None use only events with this label
        :param idx_list: list of ints or None, if set then only these indices are used for the sev creation
        :param pulse_height_intervall: list of length 2 (interval), the upper
            and lower bound for the pulse heights to include into the creation of the SEV
        :param left_right_cutoff: float, the maximal abs value of the linear slope of events
            to be included in the Sev calculation; based on the sample index as x-values
        :param rise_time_intervall: lists of length 2 (interval), the upper
            and lower bound for the rise time to include into the creation of the SEV;
            based on the sample index as x-values
        :param decay_time_intervall: list of length 2 (interval), the upper
            and lower bound for the decay time to include into the creation of the SEV;
            based on the sample index as x-values
        :param onset_intervall:  list of length 2 (interval), the upper
            and lower bound for the onset time to include into the creation of the SEV;
            based on the sample index as x-values
        :param remove_offset: bool, if True the offset is removed before the events are superposed for the
            sev calculation; highly recommended!
        :param verb: bool, if True some verbal feedback is output about the progress of the method
        :param scale_fit_height: bool, if True the parametric fit to the sev is normalized to height 1 after
            the fit is done
        :param sample_length: float, the length of one sample in milliseconds
        :return: -
        """

        h5f = h5py.File(self.path_h5, 'r+')

        if correct_label is None and idx_list is None:
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

        if idx_list is None:
            idx_list = [i for i in range(len(labels))]

        events = h5f[type]['event'][channel, idx_list, :]
        mainpar = h5f[type]['mainpar'][channel, idx_list, :]
        if labels is not None:
            labels = labels[idx_list]

        sev = h5f.require_group('stdevent_{}'.format(naming))

        sev_event, par = generate_standard_event(events=events,
                                                 main_parameters=mainpar,
                                                 labels=labels,
                                                 correct_label=correct_label,
                                                 pulse_height_intervall=pulse_height_intervall,
                                                 left_right_cutoff=left_right_cutoff,
                                                 rise_time_intervall=rise_time_intervall,
                                                 decay_time_intervall=decay_time_intervall,
                                                 onset_intervall=onset_intervall,
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

        h5f.close()


    def recalc_NPS(self, use_labels=False):
        """
        Calculates the mean Noise Power Spectrum with option to use only the baselines
        that are labeled as noise (label == 3)
        :param use_labels: bool, if True only baselines that are labeled as noise are included
        :return: -
        """

        # open file
        h5f = h5py.File(self.path_h5, 'r+')
        baselines = np.array(h5f['noise']['events'])
        labels = np.array(h5f['noise']['labels'])

        mean_nps = []
        for c in range(self.nmbr_channels):
            bl = baselines[c]
            if use_labels:
                bl = bl[labels[c] == 3]  # 3 is noise label
            mean_nps.append(calculate_mean_nps(bl)[0])

        mean_nps = np.array([mean_nps[i] for i in range(self.nmbr_channels)])

        h5f['noise'].require_dataset('nps',
                                     shape=mean_nps.shape,
                                     dtype='float')
        h5f['noise']['nps'] = mean_nps

        h5f.close()

    def recalc_additional_mp(self, type, path_h5=None, down=1):
        """
        Calculate the additional Main Parameters for the Events in an HDF5 File.

        :param type: string, either events or testpulses
        :param path_h5: string, optional, the full path to the hdf5 file, e.g. "data/bck_001.h5"
        :param down: int, the downsample rate before calculating the parameters
        :return: -
        """

        if not path_h5:
            path_h5 = self.path_h5

        h5f = h5py.File(path_h5, 'r+')
        events = h5f[type]

        of_real = np.array(h5f['optimumfilter']['optimumfilter_real'])
        of_imag = np.array(h5f['optimumfilter']['optimumfilter_real'])
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
