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
                  ):
    """
    A class for the processing of raw data events.

    The DataHandler class is one of the core parts of the cait Package. An instance of the class is bound to a HDF5 file
    and stores all data from the recorded binary files (*.rdt, ...), as well as the calculated features
    (main parameters, standard events, ...) in the file.

    :param run: The number of the measuremend run. This is a necessary argument, to identify a measurement with a given module uniquely.
    :type run: string
    :param module: The naming of the detector module. Necessary for unique identification of the physics data.
    :type module: string
    :param channels: The channels in the *.rdt file that belong to the detector module. Attention - the channel number written in the *.par file starts counting from 1, while Cait, CCS and other common software frameworks start counting from 0.
    :type channels: list of integers
    :param record_length: The number of samples in one record window. To ensure performance of all features, this should be a power of 2.
    :type record_length: int
    :param sample_frequency: The sampling frequency of the recording.
    :type sample_frequency: int

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

    def __init__(self, run: str = '01', module: str = 'Test', channels: list = [0, 1],
                 record_length: int = 16384, sample_frequency: int = 25000):
        self.run = run
        self.module = module
        self.record_length = record_length
        self.nmbr_channels = len(channels)
        self.channels = channels
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
            for c in range(len(channels)):
                self.channel_names.append('Channel ' + str(c))
                if c == len(channels) - 1 and c > 0:
                    self.colors.append('blue')
                else:
                    self.colors.append('red')

        print('DataHandler Instance created.')

    def set_filepath(self,
                     path_h5: str,
                     fname: str,
                     appendix=True):
        """
        Set the path to the *.h5 file for further processing.

        This function is usually called right after the initialization of a new object. If the intance has already done
        the conversion from *.rdt to *.h5, the path is already set automatically and the call is obsolete.

        :param path_h5: Directory that contains the H5 file.
        :type path_h5: string
        :param fname: The name of the H5 file.
        :type fname: string
        :param appendix: If true, an appendix like "-P_ChX-[...]" is automatically appended to the path_h5 string.
        :type appendix: bool

        >>> dh.set_filepath(path_h5='./', fname='test_001')
        """
        app = ''
        if appendix:
            if self.nmbr_channels == 2:
                app = '-P_Ch{}-L_Ch{}'.format(*self.channels)
            else:
                for i, c in enumerate(self.channels):
                    app += '-{}_Ch{}'.format(i + 1, c)

        # check if the channel number matches the file, otherwise error
        self.path_h5 = "{}/{}{}.h5".format(path_h5,
                                           fname, app)
        self.path_directory = path_h5
        self.fname = fname

    def import_labels(self,
                      path_labels: str,
                      type: str = 'events',
                      path_h5=None):
        """
        Include the *.csv file with the labels into the HDF5 File.

        :param path_labels: Path to the folder that contains the csv file.
            E.g. "data" looks for labels in "data/labels_bck_0XX_<type>".
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

        path_labels = '{}/labels_{}_{}.csv'.format(
            path_labels, self.fname, type)

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

    def downsample_raw_data(self, type: str = 'events', down: int = 16, dtype: str = 'float32',
                            name_appendix: str = '', delete_old:bool=True):
        """
        Downsample the dataset "event" from a specified group in the HDF5 file.

        For large scale analysis and limited server space, the covnerted HDF5 datasets exceed storage space capacities.
        For this scenario, the raw data events can be downsampled of a given factor. Downsampling to sample frequencies
        below 1kHz is in many situations sufficient for viewing events and most features calculations.

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
                          delete_old: bool=True):
        """
        Truncate the record window of the dataset "event" from a specified group in the HDF5 file.

        For measurements with high event rate (above ground, ...) a long record window might be counter productive,
        due to more pile uped events in the window. For this reason, you can truncate the length of the record window
        with this function.

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
                h5f[type].create_dataset('event'+name_appendix, data=events, dtype=dtype)
                print('New Dataset Event truncated to interval {}:{} created in group {}.'.format(truncated_idx_low,
                                                                                                  truncated_idx_up,
                                                                                                  type))
            else:
                raise FileNotFoundError('There is no event dataset in group {} in the HDF5 file.'.format(type))
