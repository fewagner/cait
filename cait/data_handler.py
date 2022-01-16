# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import os
import numpy as np
import h5py
from .mixins._data_handler_simulate import SimulateMixin
from .mixins._data_handler_rdt import RdtMixin
from .mixins._data_handler_plot import PlotMixin
from .mixins._data_handler_features import FeaturesMixin
from .mixins._data_handler_analysis import AnalysisMixin
from .mixins._data_handler_fit import FitMixin
from .mixins._data_handler_csmpl import CsmplMixin
from .mixins._data_handler_ml import MachineLearningMixin
from .mixins._data_handler_bin import BinMixin
import warnings


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler(SimulateMixin,
                  RdtMixin,
                  PlotMixin,
                  FeaturesMixin,
                  AnalysisMixin,
                  FitMixin,
                  CsmplMixin,
                  MachineLearningMixin,
                  BinMixin,
                  ):
    """
    A class for the processing of raw data events.

    The DataHandler class is one of the core parts of the cait Package. An instance of the class is bound to a HDF5 file
    and stores all data from the recorded binary files (*.rdt, ...), as well as the calculated features
    (main parameters, standard events, ...) in the file.

    :param record_length: The number of samples in one record window. To ensure performance of all features, this should be a power of 2.
    :type record_length: int
    :param sample_frequency: The sampling frequency of the recording.
    :type sample_frequency: int
    :param channels: The channels in the *.rdt file that belong to the detector module. Attention - the channel number written in the *.par file starts counting from 1, while Cait, CCS and other common software frameworks start counting from 0.
    :type channels: list of integers or None
    :param nmbr_channels: The total number of channels.
    :type nmbr_channel: int or None
    :param run: The number of the measurement run. This is a optional argument, to identify a measurement with a
        given module uniquely. Providing this argument has no effect, but might be useful in case you start multiple
        DataHandlers at once, to stay organized.
    :type run: string or None
    :param module: The naming of the detector module. Optional argument, for unique identification of the physics data.
        Providing this argument has no effect, but might be useful in case you start multiple
        DataHandlers at once, to stay organized.
    :type module: string or None

    Example for the generation of an HDF5 for events from an test *.rdt file:

    >>> import cait as ai
    >>> test_data = ai.data.TestData(filepath='test_001')
    >>> test_data.generate(source='hw')
    Rdt file written.
    Con file written.
    Par file written.
    >>> dh = ai.DataHandler()
    DataHandler instance created.
    >>> dh.convert_dataset(path_rdt='./', fname='test_001', path_h5='./')
    Start converting.
    READ EVENTS FROM RDT FILE.
    Total Records in File:  40
    Event Counts:  19
    WORKING ON EVENTS WITH TPA = 0.
    CREATE DATASET WITH EVENTS.
    CALCULATE MAIN PARAMETERS.
    WORKING ON EVENTS WITH TPA = -1.
    CREATE DATASET WITH NOISE.
    WORKING ON EVENTS WITH TPA > 0.
    CREATE DATASET WITH TESTPULSES.
    CALCULATE MP.
    Hdf5 dataset created in  ./
    Filepath and -name saved.

    Most of the methods are included via parent mixin classes (see folder cait/mixins).
    """

    def __init__(self,
                 record_length: int = 16384,
                 sample_frequency: int = 25000,
                 channels: list = None,
                 nmbr_channels: int = None,
                 run: str = None,
                 module: str = None):

        assert channels is not None or nmbr_channels is not None, 'You need to specify either the channels numbers or the number' \
                                                                  ' of channels!'

        self.run = run
        self.module = module
        self.record_length = record_length
        self.channels = channels
        if channels is not None:
            self.nmbr_channels = len(channels)
            if nmbr_channels is not None:
                warnings.warn("If channels are specified, the number of channels is taken from the length of this list,"
                              "not from the nmbr_channels argument!")
        elif nmbr_channels is not None:
            self.nmbr_channels = nmbr_channels
        self.sample_frequency = sample_frequency
        self.sample_length = 1000 / self.sample_frequency
        self.t = (np.arange(0, self.record_length, dtype=float) -
                  self.record_length / 4) * self.sample_length

        if self.nmbr_channels == 2:
            self.channel_names = ['Phonon', 'Light']
            self.colors = ['red', 'blue']
        else:
            self.channel_names = []
            self.colors = []
            for c in range(self.nmbr_channels):
                self.channel_names.append('Channel ' + str(c))
                if c == self.nmbr_channels - 1 and c > 0:
                    self.colors.append('blue')
                else:
                    self.colors.append('red')

        print('DataHandler Instance created.')

    def set_filepath(self,
                     path_h5: str,
                     fname: str,
                     appendix: bool = True,
                     channels: list = None):
        """
        Set the path to the *.h5 file for further processing.

        This function is usually called right after the initialization of a new object. If the intance has already done
        the conversion from *.rdt to *.h5, the path is already set automatically and the call is obsolete.

        :param path_h5: The path to the directory that contains the H5 file, e.g. "data/" --> file name "data/bck_001-P_Ch01-L_Ch02.csv".
        :type path_h5: string
        :param fname: The name of the H5 file.
        :type fname: string
        :param appendix: If true, an appendix like "-P_ChX-[...]" is automatically appended to the path_h5 string.
        :type appendix: bool
        :param channels: The channels in the *.rdt file that belong to the detector module. Attention - the channel number written in the *.par file starts counting from 1, while Cait, CCS and other common software frameworks start counting from 0.
        :type channels: list of integers or None

        >>> dh.set_filepath(path_h5='./', fname='test_001')
        """

        if channels is not None:
            self.channels = channels

        if path_h5 == '':
            path_h5 = './'
        if path_h5[-1] != '/':
            path_h5 = path_h5 + '/'

        app = ''
        if appendix:
            assert self.channels is not None, 'To generate the file appendix automatically, you need to specify the channel' \
                                              'numbers.'
            if self.nmbr_channels == 2:
                app = '-P_Ch{}-L_Ch{}'.format(*self.channels)
            else:
                for i, c in enumerate(self.channels):
                    app += '-{}_Ch{}'.format(i + 1, c)

        # check if the channel number matches the file, otherwise error
        self.path_h5 = "{}{}{}.h5".format(path_h5, fname, app)
        self.path_directory = path_h5
        self.fname = fname

    def import_labels(self,
                      path_labels: str,
                      type: str = 'events',
                      path_h5=None):
        """
        Include the *.csv file with the labels into the HDF5 File.

        :param path_labels: Path to the folder that contains the csv file.
            E.g. "data/" looks for labels in "data/labels_bck_0XX_<type>".
        :type path_labels: string
        :param type: The group name in the HDF5 file of the events, typically "events" or "testpulses".
        :type type: string
        :param path_h5: Provide an alternative full path to the HDF5 file to include the labels,
            e.g. "data/hdf5s/bck_001[...].h5".
        :type path_h5: string or None

        >>> ei = ai.EventInterface()
        Event Interface Instance created.
        >>> ei.load_h5(path='./',fname='test_001',channels=[0,1])
        Nmbr triggered events:  4
        Nmbr testpulses:  11
        Nmbr noise:  4
        Bck File loaded.
        >>> ei.create_labels_csv(path='./')
        >>> dh.import_labels(path_labels='./')
        Added Labels.
        """

        if not path_h5:
            path_h5 = self.path_h5

        if path_labels == '':
            path_labels = './'
        if path_labels[-1] != '/':
            path_labels = path_labels + '/'
        path_labels = '{}labels_{}_{}.csv'.format(path_labels, self.fname,
                                                  type)

        with h5py.File(path_h5, 'r+') as h5f:

            if path_labels != '' and os.path.isfile(path_labels):
                labels = np.genfromtxt(path_labels)
                labels = labels.astype('int32')
                length = len(labels)
                labels.resize(
                    (self.nmbr_channels, int(length / self.nmbr_channels)))

                events = h5f[type]

                if "labels" in events:
                    events['labels'][...] = labels
                    print('Edited Labels.')

                else:
                    events.create_dataset('labels', data=labels)
                    events['labels'].attrs.create(name='unlabeled', data=0)
                    events['labels'].attrs.create(name='Event_Pulse', data=1)
                    events['labels'].attrs.create(name='Test/Control_Pulse', data=2)
                    events['labels'].attrs.create(name='Noise', data=3)
                    events['labels'].attrs.create(name='Squid_Jump', data=4)
                    events['labels'].attrs.create(name='Spike', data=5)
                    events['labels'].attrs.create(name='Early_or_late_Trigger', data=6)
                    events['labels'].attrs.create(name='Pile_Up', data=7)
                    events['labels'].attrs.create(name='Carrier_Event', data=8)
                    events['labels'].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
                    events['labels'].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
                    events['labels'].attrs.create(name='Decaying_Baseline', data=11)
                    events['labels'].attrs.create(name='Temperature_Rise', data=12)
                    events['labels'].attrs.create(name='Stick_Event', data=13)
                    events['labels'].attrs.create(name='Square_Waves', data=14)
                    events['labels'].attrs.create(name='Human_Disturbance', data=15)
                    events['labels'].attrs.create(name='Large_Sawtooth', data=16)
                    events['labels'].attrs.create(name='Cosinus_Tail', data=17)
                    events['labels'].attrs.create(name='Light_only_Event', data=18)
                    events['labels'].attrs.create(name='Ring_Light_Event', data=19)
                    events['labels'].attrs.create(
                        name='Sharp_Light_Event', data=20)
                    events['labels'].attrs.create(name='unknown/other', data=99)

                    print('Added Labels.')

            elif (path_labels != ''):
                print("File '{}' does not exist.".format(path_labels))

    def import_predictions(self,
                           model: str,
                           path_predictions: str,
                           type: str = 'events',
                           only_channel: int = None,
                           path_h5: str = None):
        """
        Include the *.csv file with the predictions from a machine learning model into the HDF5 File.

        :param model: The naming for the type of model, e.g. Random Forest --> "RF".
        :type model: string
        :param path_predictions: Path to the folder that contains the csv file.
            E.g. "data/" --> look for predictions in "data/<model>_predictions_<self.fname>_<type>".
            If the argument only_channel is not None, then additionally "_Ch<only_channel>" is append to the
            looked for file.
        :type path_predictions: string
        :param type: The name of the group in the HDF5 file, typically "events" or "testpulses".
        :type type: string
        :param only_channel: If the labels are only for a specific channel then define here for which channel.
        :type only_channel: int or None
        :param path_h5: Provide an alternative (full) path to the HDF5 file, e.g. "data/hdf5s/bck_001[...].h5".
        :type path_h5: string or None

        >>> dh.import_predictions(model='RF', path_predictions='./')
        Added RF Predictions.
        """

        if not path_h5:
            path_h5 = self.path_h5

        if only_channel is not None:
            app = '_Ch{}'.format(only_channel)
        else:
            app = ''

        if path_predictions == '':
            path_predictions = './'
        if path_predictions[-1] != '/':
            path_predictions = path_predictions + '/'
        path_predictions = '{}{}_predictions_{}_{}{}.csv'.format(
            path_predictions, model, self.fname, type, app)

        with h5py.File(path_h5, 'r+') as h5f:
            events = h5f[type]

            if path_predictions != '' and os.path.isfile(path_predictions):
                labels = np.genfromtxt(path_predictions)
                labels = labels.astype('int32')
                length = len(labels)
                if only_channel is None:
                    labels.resize(
                        (self.nmbr_channels, int(length / self.nmbr_channels)))
                # not overwrite the other channels
                elif "{}_predictions".format(model) in events:
                    labels_full = np.array(
                        events["{}_predictions".format(model)][...])
                    labels_full[only_channel, :] = labels[:]
                    labels = np.copy(labels_full)
                    del labels_full  # free the memory of the dummy array again
                else:  # overwrite the other channels with zeros
                    labels_full = np.zeros([self.nmbr_channels, length])
                    labels_full[only_channel, :] = labels
                    labels = np.copy(labels_full)
                    del labels_full  # free the memory of the dummy array again

                if "{}_predictions".format(model) in events:
                    events["{}_predictions".format(model)][...] = labels
                    print('Edited Predictions.')

                else:
                    events.create_dataset(
                        "{}_predictions".format(model), data=labels)
                    events["{}_predictions".format(model)].attrs.create(
                        name='unlabeled', data=0)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Event_Pulse', data=1)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Test/Control_Pulse', data=2)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Noise', data=3)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Squid_Jump', data=4)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Spike', data=5)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Early_or_late_Trigger', data=6)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Pile_Up', data=7)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Carrier_Event', data=8)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Strongly_Saturated_Event_Pulse', data=9)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Strongly_Saturated_Test/Control_Pulse', data=10)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Decaying_Baseline', data=11)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Temperature_Rise', data=12)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Stick_Event', data=13)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Sawtooth_Cycle', data=14)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Human_Disturbance', data=15)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Large_Sawtooth', data=16)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Cosinus_Tail', data=17)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Light_only_Event', data=18)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Ring_Light_Event', data=19)
                    events["{}_predictions".format(model)].attrs.create(
                        name='Sharp_Light_Event', data=20)
                    events["{}_predictions".format(model)].attrs.create(
                        name='unknown/other', data=99)

                    print('Added {} Predictions.'.format(model))

            else:
                raise KeyError(
                    'No prediction file found at {}.'.format(path_predictions))

    def get_filehandle(self, path=None):
        """
        Get the opened filestream to the HDF5 file.

        This is usually needed for individual feature calculations, plots or cuts, that are not ready-to-play implemented.

        :param path: Provide an alternative full path to the HDF5 file of that we want to open the file stream.
        :type path: string or None
        :return: The opened file stream. Please look into the h5py Python library for details about the file stream.
        :rtype: h5py file stream

        >>> with dh.get_filehandle() as f:
        ...     f.keys()
        ...
        <KeysViewHDF5 ['events', 'noise', 'testpulses']>
        """
        if path is None:
            f = h5py.File(self.path_h5, 'r+')
        else:
            f = h5py.File(path, 'r+')
        return f

    def drop_raw_data(self, type: str = 'events'):
        """
        Delete the dataset "event" from a specified group in the HDF5 file.

        For large scale analysis and limited server space, the covnerted HDF5 datasets exceed storage space capacities.
        For this scenario, the raw data events can be deleted after the calculation of all useful features. At a later
        point, the events can be included again if needed.

        Attention, due to the tree structure of the HDF5 file, this will not actually reduce the size!
        For reducing the file size, the HDF5 file has to be repacked with the h5repack method of the HDF5 Tools,
        see https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack. This can be done on Ubuntu/Mac e.g. with

        >>> h5repack test_data/test_001.h5 test_data/test_001_copy.h5
        >>> rm test_data/test_001.h5
        >>> mv test_data/test_001_copy.h5 test_data/test_001.h5

        :param type: The group in the HDF5 set from which the events are deleted,
            typically 'events', 'testpulses' or noise.
        :type type: string

        >>> dh.drop_raw_data()
        Dataset Event deleted from group events.
        """
        with h5py.File(self.path_h5, 'r+') as h5f:
            if "event" in h5f[type]:
                del h5f[type]['event']
                print('Dataset Event deleted from group {}.'.format(type))
            else:
                raise FileNotFoundError('There is no event dataset in group {} in the HDF5 file.'.format(type))

    def drop(self, group: str, dataset: str = None):
        """
        Delete a dataset from a specified group in the HDF5 file.

        Attention, due to the tree structure of the HDF5 file, this will not actually reduce the size!
        For reducing the file size, the HDF5 file has to be repacked with the h5repack method of the HDF5 Tools,
        see https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack. This can be done on Ubuntu/Mac e.g. with

        >>> h5repack test_data/test_001.h5 test_data/test_001_copy.h5
        >>> rm test_data/test_001.h5
        >>> mv test_data/test_001_copy.h5 test_data/test_001.h5

        :param group: The name of the group in the HDF5 file.
        :type group: string
        :param dataset: The name of the dataset in the HDF5 file. If None, the would group is deleted.
        :type dataset: string
        """

        with h5py.File(self.path_h5, 'r+') as h5f:
            if dataset in h5f[group]:
                del h5f[group][dataset]
                print('Dataset {} deleted from group {}.'.format(dataset, group))
            else:
                raise FileNotFoundError('There is no dataset {} in group {} in the HDF5 file.'.format(dataset, group))

    def downsample_raw_data(self, type: str = 'events', down: int = 16, dtype: str = 'float32',
                            name_appendix: str = '', delete_old: bool = True):
        """
        Downsample the dataset "event" from a specified group in the HDF5 file.

        For large scale analysis and limited server space, the covnerted HDF5 datasets exceed storage space capacities.
        For this scenario, the raw data events can be downsampled of a given factor. Downsampling to sample frequencies
        below 1kHz is in many situations sufficient for viewing events and most features calculations.

        Attention, due to the tree structure of the HDF5 file, this will not actually reduce the size!
        For reducing the file size, the HDF5 file has to be repacked with the h5repack method of the HDF5 Tools,
        see https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack. This can be done on Ubuntu/Mac e.g. with

        >>> h5repack test_data/test_001.h5 test_data/test_001_copy.h5
        >>> rm test_data/test_001.h5
        >>> mv test_data/test_001_copy.h5 test_data/test_001.h5

        :param type: The group in the HDF5 set from which the events are downsampled,
            typically 'events', 'testpulses' or noise.
        :type type: string
        :param down: The factor by which the data is downsampled. This should be a factor of 2.
        :type down: int
        :param dtype: The data type of the new stored events, typically you want this to be float32.
        :type dtype: string
        :param name_appendix: An appendix to the dataset event in order to keep the old and the new events.
        :type name_appendix: string
        :param delete_old: If true, the old events are deleted. Deactivate only, if an unique name_appendix is choosen.
        :type delete_old: bool

        >>> dh.downsample_raw_data()
        Old Dataset Event deleted from group events.
        New Dataset Event with downsample rate 16 created in group events.
        """

        if name_appendix == '' and not delete_old:
            raise KeyError('To keep the old events please choose appropriate name appendix for the new!')

        with h5py.File(self.path_h5, 'r+') as h5f:
            if "event" in h5f[type]:
                events = np.array(h5f[type]['event'])
                if delete_old:
                    del h5f[type]['event']
                print('Old Dataset Event deleted from group {}.'.format(type))
                events = np.mean(events.reshape((events.shape[0], events.shape[1], int(events.shape[2] / down), down)),
                                 axis=3)
                h5f[type].create_dataset('event' + name_appendix, data=events, dtype=dtype)
                print('New Dataset Event with downsample rate {} created in group {}.'.format(down, type))
            else:
                raise FileNotFoundError('There is no event dataset in group {} in the HDF5 file.'.format(type))

    def truncate_raw_data(self, type: str,
                          truncated_idx_low: int,
                          truncated_idx_up: int,
                          dtype: str = 'float32',
                          name_appendix: str = '',
                          delete_old: bool = True):
        """
        Truncate the record window of the dataset "event" from a specified group in the HDF5 file.

        For measurements with high event rate (above ground, ...) a long record window might be counter productive,
        due to more pile uped events in the window. For this reason, you can truncate the length of the record window
        with this function.

        Attention, due to the tree structure of the HDF5 file, this will not actually reduce the size!
        For reducing the file size, the HDF5 file has to be repacked with the h5repack method of the HDF5 Tools,
        see https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack. This can be done on Ubuntu/Mac e.g. with

        >>> h5repack test_data/test_001.h5 test_data/test_001_copy.h5
        >>> rm test_data/test_001.h5
        >>> mv test_data/test_001_copy.h5 test_data/test_001.h5

        :param type: The group in the HDF5 set from which the events are downsampled,
            typically 'events', 'testpulses' or noise.
        :type type: string
        :param truncated_idx_low: The lower index within the old record window, that becomes the first index in the
            truncated record window.
        :type truncated_idx_low: int
        :param truncated_idx_up: The upper index winthin the old record window, that becomes the last index in the
            truncated record window.
        :type truncated_idx_up: int
        :param dtype: The data type of the new stored events, typically you want this to be float32.
        :type dtype: string
        :param name_appendix: An appendix to the dataset event in order to keep the old and the new events.
        :type name_appendix: string
        :param delete_old: If true, the old events are deleted. Deactivate only, if an unique name_appendix is choosen.
        :type delete_old: bool
        """

        if name_appendix == '' and not delete_old:
            raise KeyError('To keep the old events please choose appropriate name appendix for the new!')

        with h5py.File(self.path_h5, 'r+') as h5f:
            if "event" in h5f[type]:
                events = np.array(h5f[type]['event'])
                if delete_old:
                    del h5f[type]['event']
                print('Old Dataset Event deleted from group {}.'.format(type))
                events = events[:, :, truncated_idx_low:truncated_idx_up]
                h5f[type].create_dataset('event' + name_appendix, data=events, dtype=dtype)
                print('New Dataset Event truncated to interval {}:{} created in group {}.'.format(truncated_idx_low,
                                                                                                  truncated_idx_up,
                                                                                                  type))
            else:
                raise FileNotFoundError('There is no event dataset in group {} in the HDF5 file.'.format(type))

    def get(self, group: str, dataset: str):
        """
        Get a dataset from the HDF5 file with save closing of the file stream.

        :param group: The name of the group in the HDF5 set.
        :type group: string
        :param dataset: The name of the dataset in the HDF5 set. There are special key word for calculated properties
            from the main parameters, namely 'pulse_height', 'onset', 'rise_time', 'decay_time', 'slope'. These are
            consistent with used in the cut when generating a standard event.
        :type dataset: string
        :return: The dataset from the HDF5 file
        :rtype: numpy array
        """
        add_mainpar_names = ['array_max', 'array_min', 'var_first_eight', 'mean_first_eight', 'var_last_eight',
                             'mean_last_eight', 'var', 'mean', 'skewness', 'max_derivative', 'ind_max_derivative',
                             'min_derivative', 'ind_min_derivative', 'max_filtered', 'ind_max_filtered',
                             'skewness_filtered_peak']

        with h5py.File(self.path_h5, 'r') as f:
            if dataset == 'pulse_height' and 'pulse_height' not in f[group]:
                data = np.array(f[group]['mainpar'][:, :, 0])
            elif dataset == 'onset':
                data = np.array((f[group]['mainpar'][:, :, 1] - self.record_length / 4) / self.sample_frequency * 1000)
            elif dataset == 'rise_time':
                data = np.array(
                    (f[group]['mainpar'][:, :, 2] - f[group]['mainpar'][:, :, 1]) / self.sample_frequency * 1000)
            elif dataset == 'decay_time':
                data = np.array(
                    (f[group]['mainpar'][:, :, 6] - f[group]['mainpar'][:, :, 4]) / self.sample_frequency * 1000)
            elif dataset == 'slope':
                data = np.array(f[group]['mainpar'][:, :, 8] * self.record_length)
            else:
                for i, name in enumerate(add_mainpar_names):
                    if dataset == name and name not in f[group]:
                        data = np.array(f[group]['add_mainpar'][:, :, i])
                        break
                else:
                    data = np.array(f[group][dataset])
        return data

    def keys(self, group: str = None):
        """
        Print the keys of the HDF5 file or a group within it.

        :param group: The name of a group in the HDF5 file of that we print the keys.
        :type group: string or None
        """
        with h5py.File(self.path_h5, 'r+') as f:
            if group is None:
                print(list(f.keys()))
            else:
                print(list(f[group].keys()))

    def content(self):
        """
        Print the whole content of the HDF5 and all derived properties.
        """
        print('The following properties are in the HDF5 sets can be accessed through the get(group, dataset) methode.')

        with h5py.File(self.path_h5, 'r+') as f:
            for group in f.keys():
                print(f'The following data sets are contained in the the group {group}:')
                for dataset in f[group].keys():
                    print(f'dataset: {dataset}, shape: {f[group][dataset].shape}')
                if 'mainpar' in f[group].keys():
                    shape = f[group]['mainpar'].shape[:2]
                    for dataset in ['pulse_height', 'onset', 'rise_time', 'decay_time', 'slope']:
                        print(f'dataset: {dataset}, shape: {shape}')
                if 'add_mainpar' in f[group].keys():
                    shape = f[group]['add_mainpar'].shape[:2]
                    for dataset in ['array_max', 'array_min', 'var_first_eight', 'mean_first_eight', 'var_last_eight',
                                    'mean_last_eight', 'var', 'mean', 'skewness', 'max_derivative',
                                    'ind_max_derivative',
                                    'min_derivative', 'ind_min_derivative', 'max_filtered', 'ind_max_filtered',
                                    'skewness_filtered_peak']:
                        print(f'dataset: {dataset}, shape: {shape}')

    def generate_startstop(self):
        """
        Generate a startstop data set in the metainfo group from the testpulses time stamps.
        """

        print('Generating Start Stop Metainfo.')

        with h5py.File(self.path_h5, 'r+') as f:

            assert 'testpulses' in f, 'No testpulses in file!'

            if 'origin' in f['testpulses']:
                origin = []
                first_idx = []
                last_idx = []
                for i, fname in enumerate(f['testpulses']['origin']):
                    if fname not in origin:
                        origin.append(fname)
                        first_idx.append(i)
                        last_idx.append(i - 1)

                del last_idx[0]
                last_idx.append(len(f['testpulses']['origin']) - 1)

                origin = [name.decode('UTF-8') for name in origin]  # name.encode().decode('UTF-8')

                print('Unique Files: ', origin)

                startstop_hours = np.empty((len(origin), 2), dtype=float)
                startstop_s = np.empty((len(origin), 2), dtype=int)
                startstop_mus = np.empty((len(origin), 2), dtype=int)

                for i, fname in enumerate(origin):
                    startstop_hours[i, :] = (f['testpulses']['hours'][first_idx[i]],
                                             f['testpulses']['hours'][last_idx[i]])
                    startstop_s[i, :] = (f['testpulses']['time_s'][first_idx[i]],
                                         f['testpulses']['time_s'][last_idx[i]])
                    startstop_mus[i, :] = (f['testpulses']['time_mus'][first_idx[i]],
                                           f['testpulses']['time_mus'][last_idx[i]])

            else:
                print('One unique file detected.')

                startstop_hours = np.array([[f['testpulses']['hours'][0],
                                             f['testpulses']['hours'][-1]]])
                startstop_s = np.array([[f['testpulses']['time_s'][0],
                                         f['testpulses']['time_s'][-1]]])
                startstop_mus = np.array([[f['testpulses']['time_mus'][0],
                                           f['testpulses']['time_mus'][-1]]])

            metainfo = f.require_group('metainfo')

            datasets = ["startstop_hours", "startstop_s", "startstop_mus"]
            if 'origin' in f['testpulses']:  # comment, error!
                datasets.append("origin")

            for name in datasets:
                if name in metainfo:
                    del metainfo[name]
                metainfo.create_dataset(name, data=eval(name))

    def init_empty(self):
        """
        Initialize an empty HDF5 set.
        """
        with h5py.File(self.path_h5, 'a') as h5f:
            pass

    def record_window(self, ms=True):
        """
        Get the t array corresponding to a typical record window.

        :param ms: If true, the time is in ms. Otherwise in s.
        :type ms: bool
        :return: the time array.
        :rtype: 1D numpy array
        """
        t = (np.arange(self.record_length) - self.record_length/4)/self.sample_frequency
        if ms:
            t *= 1000
        return t
