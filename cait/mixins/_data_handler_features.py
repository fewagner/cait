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
from sklearn.decomposition import PCA

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
    def recalc_sev(self,
                   type='events',
                   use_labels=False,
                   correct_label=None,
                   use_idx=None,
                   pulse_height_interval=None,
                   left_right_cutoff=None,
                   rise_time_interval=None,
                   decay_time_interval=None,
                   onset_interval=None,
                   remove_offset=True,
                   verb=True,
                   scale_fit_height=True,
                   sample_length=0.04):
        """
        Calculate the Standard Event for the Events in the HDF5 File.

        :param type: string, either "events" or "testpulses"
        :param use_labels: bool, if True a labels file must be included in the hdf5 file,
            then only the events labeled as events or testpulses are included in the calculation
        :param correct_label: int, the label to be used for the sev generation
        :param use_idx: list of ints, only these indices are included for the sev generation
        :param pulse_height_interval: list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the pulse heights to include into the creation of the SEV
        :param left_right_cutoff: list of NMBR_CHANNELS floats, the maximal abs value of the linear slope of events
            to be included in the Sev calculation; based on the sample index as x-values
        :param rise_time_interval: list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the rise time to include into the creation of the SEV;
            based on the sample index as x-values
        :param decay_time_interval: list of NMBR_CHANNELS lists of length 2 (intervals), the upper
            and lower bound for the decay time to include into the creation of the SEV;
            based on the sample index as x-values
        :param onset_interval:  list of NMBR_CHANNELS lists of length 2 (intervals), the upper
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

        # if no pulse_height_interval is specified set it to average values for all channels
        if pulse_height_interval == None:
            pulse_height_interval = [[0.5, 1.5]
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
                sev = h5f.require_group('stdevent')
                correct_label = 1
            elif type == 'testpulses':
                sev = h5f.require_group('stdevent_tp')
                correct_label = 2
            else:
                raise NotImplementedError('Type must be events or testpulses!')
        else:
            sev = h5f.require_group('stdevent_{}'.format(correct_label))

        if use_idx is None:
            use_idx = list(range(len(events[0])))

        for c in range(self.nmbr_channels):
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

    def recalc_of(self, down=1):
        """
        Calculate the Optimum Filer from the NPS and the SEV

        :return: -
        """

        h5f = h5py.File(self.path_h5, 'r+')
        stdevent_pulse = [h5f['stdevent']['event'][i]
                          for i in range(self.nmbr_channels)]
        mean_nps = [h5f['noise']['nps'][i] for i in range(self.nmbr_channels)]

        print('CREATE OPTIMUM FILTER.')

        of = np.array([optimal_transfer_function(
            stdevent_pulse[i], mean_nps[i]) for i in range(self.nmbr_channels)])

        optimumfilter = h5f.require_group('optimumfilter')
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

        h5f.close()

    # apply the optimum filter
    def apply_of(self, type='events', chunk_size=10000, hard_restrict=False):
        """
        Calculates the height of events or testpulses after applying the optimum filter

        :param type: string, either events of testpulses
        :param chunk_size: int, the size how many events are processes simultaneoursly to avoid memory error
        :param hard_restrict: bool, if True, the maximum search is restricted to 20-30% of the record window.
        :return: -
        """

        print('Calculating OF Heights.')

        f = h5py.File(self.path_h5, 'r+')
        events = f[type]['event']
        sev = np.array(f['stdevent']['event'])
        nps = np.array(f['noise']['nps'])
        of = np.zeros((self.nmbr_channels, int(self.record_length / 2 + 1)), dtype=complex)
        of.real = f['optimumfilter']['optimumfilter_real']
        of.imag = f['optimumfilter']['optimumfilter_imag']

        f[type].require_dataset(name='of_ph',
                                shape=(self.nmbr_channels, len(events[0])),
                                dtype='float')

        nmbr_events = len(events[0])
        counter = 0
        while counter + chunk_size < nmbr_events:
            for c in range(self.nmbr_channels):
                of_ph = get_amplitudes(events[c, counter:counter + chunk_size], sev[c], nps[c],
                                       hard_restrict=hard_restrict)
                f[type]['of_ph'][c, counter:counter + chunk_size] = of_ph
            counter += chunk_size
        for c in range(self.nmbr_channels):
            of_ph = get_amplitudes(events[c, counter:nmbr_events], sev[c], nps[c], hard_restrict=hard_restrict)
            f[type]['of_ph'][c, counter:nmbr_events] = of_ph

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
                             pulse_height_interval=[0.5, 1.5],
                             left_right_cutoff=None,
                             rise_time_interval=None,
                             decay_time_interval=None,
                             onset_interval=None,
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
        :param pulse_height_interval: list of length 2 (interval), the upper
            and lower bound for the pulse heights to include into the creation of the SEV
        :param left_right_cutoff: float, the maximal abs value of the linear slope of events
            to be included in the Sev calculation; based on the sample index as x-values
        :param rise_time_interval: lists of length 2 (interval), the upper
            and lower bound for the rise time to include into the creation of the SEV;
            based on the sample index as x-values
        :param decay_time_interval: list of length 2 (interval), the upper
            and lower bound for the decay time to include into the creation of the SEV;
            based on the sample index as x-values
        :param onset_interval:  list of length 2 (interval), the upper
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

        h5f.close()

    def recalc_NPS(self, use_labels=False, down=1):
        """
        Calculates the mean Noise Power Spectrum with option to use only the baselines
        that are labeled as noise (label == 3)
        :param use_labels: bool, if True only baselines that are labeled as noise are included
        :param down: int, a factor by that the baselines are downsampled before the calculation - must be 2^x
        :return: -
        """
        print('Calculate NPS.')

        # open file
        h5f = h5py.File(self.path_h5, 'r+')
        baselines = np.array(h5f['noise']['event'])
        if use_labels:
            labels = np.array(h5f['noise']['labels'])

        mean_nps = []
        for c in range(self.nmbr_channels):
            bl = baselines[c]
            if use_labels:
                bl = bl[labels[c] == 3]  # 3 is noise label
            mean_nps.append(calculate_mean_nps(bl, down=down)[0])

        mean_nps = np.array([mean_nps[i] for i in range(self.nmbr_channels)])

        naming = 'nps'
        if down > 1:
            naming += '_down' + str(down)

        h5f['noise'].require_dataset(naming,
                                     shape=mean_nps.shape,
                                     dtype='float')
        h5f['noise'][naming][...] = mean_nps

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

    def apply_pca(self, nmbr_components, type='events'):
        """
        TODO

        :param nmbr_components:
        :type nmbr_components:
        :param type:
        :type type:
        :return:
        :rtype:
        """

        f = h5py.File(self.path_h5, 'r+')
        if 'pca_projection' in f[type]:
            print('Overwrite old pca projections')
            del f[type]['pca_projection']
        if 'pca_components' in f[type]:
            print('Overwrite old pca components')
            del f[type]['pca_components']
        pca_projection = f[type].create_dataset(name='pca_projection',
                                                shape=(self.nmbr_channels, len(f['events']['hours']), nmbr_components),
                                                dtype=float)
        pca_error = f[type].require_dataset(name='pca_error',
                                            shape=(self.nmbr_channels, len(f['events']['hours'])),
                                            dtype=float)
        pca_components = f[type].create_dataset(name='pca_components',
                                                shape=(
                                                self.nmbr_channels, nmbr_components, len(f[type]['event'][0, 0])),
                                                dtype=float)

        for c in range(self.nmbr_channels):
            print('Channel ', c)
            X = f[type]['event'][c]
            X -= np.mean(X[:, :int(self.record_length / 8)], axis=1, keepdims=True)
            pca = PCA(n_components=nmbr_components)
            X_transformed = pca.fit_transform(X)

            print('Explained Variance: ', pca.explained_variance_ratio_)
            print('Singular Values: ', pca.singular_values_)

            # print('X_transformed: ', X_transformed)
            # print('PCA error: ', np.mean((pca.inverse_transform(X_transformed) - X)**2, axis=1))

            pca_projection[c, ...] = X_transformed
            pca_error[c, ...] = np.mean((pca.inverse_transform(X_transformed) - X) ** 2, axis=1)

            for i in range(nmbr_components):
                transformed = np.zeros(nmbr_components)
                transformed[i] = 1
                comp = pca.inverse_transform(transformed.reshape(1, -1))
                pca_components[c, i, :] = comp.reshape(-1)
