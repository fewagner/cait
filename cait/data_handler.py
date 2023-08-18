# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import os
import subprocess
import numpy as np
import h5py
from typing import List, Union
from .mixins._data_handler_simulate import SimulateMixin
from .mixins._data_handler_rdt import RdtMixin
from .mixins._data_handler_plot import PlotMixin
from .mixins._data_handler_features import FeaturesMixin
from .mixins._data_handler_analysis import AnalysisMixin
from .mixins._data_handler_fit import FitMixin
from .mixins._data_handler_csmpl import CsmplMixin
from .mixins._data_handler_ml import MachineLearningMixin
from .mixins._data_handler_bin import BinMixin
from .styles._print_styles import fmt_gr, fmt_ds, fmt_virt, sizeof_fmt, txt_fmt, datetime_fmt
from .versatile.file import EventIterator, ds_source_available
import warnings

MAINPAR = ['pulse_height', 'onset', 'rise_time', 'decay_time', 'slope']
ADD_MAINPAR = ['array_max', 'array_min', 'var_first_eight', 
               'mean_first_eight', 'var_last_eight', 'mean_last_eight', 
               'var', 'mean', 'skewness', 'max_derivative',
               'ind_max_derivative', 'min_derivative', 'ind_min_derivative', 
               'max_filtered', 'ind_max_filtered', 'skewness_filtered_peak']

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
                 module: str = None,
                 ):

        assert channels is not None or nmbr_channels is not None, 'You need to specify either the channels numbers or the number of channels!'

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

    def __str__(self):
        # Info on size, file, groups and possibly connected virtual datasets
        size = sizeof_fmt(os.path.getsize(self.get_filepath()))
        info = f"DataHandler linked to HDF5 file '{self.get_filepath()}'\n"
        info += f"HDF5 file size on disk: {size}\n"
        n_virtual = 0
        available = True
        external_files = set()

        with self.get_filehandle(mode="r") as f:
            groups = list(f.keys())
            for group in groups:
                for ds in f[group]:
                    if not ds_source_available(f,group,ds): available = False

                    if f[group][ds].is_virtual:
                        n_virtual+=1
                        filenames = [x[1] for x in f[group][ds].virtual_sources()]
                        external_files = external_files.union(filenames)
        
        info+= f"Groups in file: {groups}.\n\n"

        if n_virtual > 0:
            info += f"The HDF5 file contains virtual datasets linked to the following files: {external_files}\n"
            if available:
                info += f"All of the external sources are currently available.\n\n"
            else:
                info += f"{txt_fmt('Some of the external sources are currently unavailable.', 'red', 'bold')}\n\n"

        # Info on start/end/length, if available
        with self.get_filehandle(mode="r") as f:
            if "testpulses" in f.keys():
                if "hours" in f["testpulses"]:
                    total_time = f["testpulses/hours"][-1] - f["testpulses/hours"][0]
                    info += f"Time between first and last testpulse: {total_time:.2f} h\n"
                if "time_s" in f["testpulses"]:
                    info += f"First testpulse on/at: {datetime_fmt(f['testpulses/time_s'][0])}\n"
                    info += f"Last testpulse on/at: {datetime_fmt(f['testpulses/time_s'][-1])}\n"

        return info
        
    def set_filepath(self,
                     path_h5: str,
                     fname: str,
                     appendix: bool = True,
                     channels: list = None):
        """
        Set the path to the *.h5 file for further processing.

        This function is usually called right after the initialization of a new object. If the instance has already done
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

        if channels is not None: self.channels = channels

        app = ''
        if appendix:
            assert self.channels is not None, 'To generate the file appendix automatically, you need to specify the channel numbers.'
            if self.nmbr_channels == 2:
                app = '-P_Ch{}-L_Ch{}'.format(*self.channels)
            else:
                for i, c in enumerate(self.channels):
                    app += '-{}_Ch{}'.format(i + 1, c)

        self.fname = fname
        self.path_directory = path_h5
        self.path_h5 = os.path.join(self.path_directory, self.fname + app + ".h5")

        if not os.path.exists(self.path_h5):
            print(f"{self.path_h5} does not exist. Use dh.init_empty() if you mean to initialize a new (empty) file.")

    def get_filepath(self, absolute: bool = False):
        """
        Get the relative path to the HDF5 file assigned to this instance of DataHandler.

        :param absolute: If true, the absolute path is returned instead.
        :type absolute: bool

        :raises Exception: If filepath has not yet been set (using DataHandler.set_filepath()).
        :raises FileNotFoundError: If the HDF5 file corresponding to the set filepath does not exist. In such a chase, an empty HDF5 file can be created using DataHandler.init_empty().

        :return: Path to the file connected to this DataHandler.
        :rtype: str
        """
        if not hasattr(self, "path_h5"): 
            raise Exception("Filepath has not been set. Use dh.set_filepath() first.")
        if not os.path.exists(self.path_h5):
            raise FileNotFoundError(f"{self.path_h5} does not exist. Use dh.init_empty() to initialize an empty HDF5 file if that is what you intend.")

        return os.path.abspath(self.path_h5) if absolute else os.path.relpath(self.path_h5)

    def get_filedirectory(self, absolute: bool = False):
        """
        Get the relative path to the directory where the HDF5 file assigned to this instance of DataHandler is stored.

        :param absolute: If true, the absolute path is returned instead.
        :type absolute: bool

        :return: Path to the directory of the HDF5 file.
        :rtype: str
        """
        return os.path.dirname(self.get_filepath(absolute))

    def get_filename(self):
        """
        Get name of the HDF5 file assigned to this instance of DataHandler.

        :param absolute: If true, the absolute path is returned instead.
        :type absolute: bool

        :return: Name of the HDF5 file (without *.h5 extension).
        :rtype: str
        """
        return os.path.splitext(os.path.basename(self.get_filepath()))[0]
    
    def get_filehandle(self, path: str = None, mode: str = "r+"):
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
                f = h5py.File(self.get_filepath(), mode)
            else:
                f = h5py.File(path, mode)
            return f
    
    def get_event_iterator(self, group: str, channel: int = None, flag: List[bool] = None, batch_size: int = None):
        """
        Returns EventIterator object that can be used to iterate events of a given group and channel. When used within a with statement, the corresponding HDF5 file is kept open for faster access.

        :param group: The name of the group in the HDF5 file.
        :type group: string
        :param channel: The channel to use. Defaults to None, which means "all channels"
        :type channel: int
        :param flag: A boolean flag of events to include in the iterator
        :type flag: list of bool

        :return: EventIterator
        :rtype: Context Manager / Iterator

        >>> # Usage as regular iterator (HDF5 file is separately opened/closed for each event)
        >>> ev_it = dh.get_event_iterator("events", 0)
        >>> for ev in ev_it:
        ...    print(np.max(ev))

        >>> # Usage as context manager (HDF5 file is kept open)

        >>> with dh.get_event_iterator("events", 0) as ev_it:
        ...     for ev in ev_it:
        ...         print(np.max(ev))
        """
        # Use the first channel and the first datapoint of a voltage trace to get the total number of events
        inds = np.arange(self.get(group, "event", 0, None, 0).size)

        if flag is not None: inds = inds[flag]

        return EventIterator(path_h5=self.get_filepath(), group=group, dataset="event", channels=channel, inds=inds, batch_size=batch_size)
    
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
            path_h5 = self.get_filepath()

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
            path_h5 = self.get_filepath()

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
    
    def drop(self, group: str, dataset: str = None, repackage: bool = False):
        """
        Delete a dataset from a specified group in the HDF5 file. If no dataset is provided, the entire group is deleted.

        Attention: Without repackaging, this method does NOT decrease the HDF5 file's size! 
        See :func:`cait.DataHandler.repackage` for details.

        :param group: The name of the group in the HDF5 file.
        :type group: string
        :param dataset: The name of the dataset in the HDF5 file. If None, the would group is deleted.
        :type dataset: string
        :param repackage: If set to True, the HDF5 file will be repackaged.
        :type repackage: bool
        """

        with h5py.File(self.get_filepath(), 'r+') as h5f:
            if dataset is None:
                del h5f[group]
                print(f'Group {fmt_gr(group)} deleted.')
            elif dataset in h5f[group]:
                del h5f[group][dataset]
                print(f'Dataset {fmt_ds(dataset)} deleted from group {fmt_gr(group)}.')
            else:
                raise FileNotFoundError('There is no dataset {} in group {} in the HDF5 file.'.format(dataset, group))
            
        if repackage: self.repackage()

    def drop_raw_data(self, type: str = 'events', repackage: bool = False):
        """
        Delete the dataset "event" from a specified group in the HDF5 file.

        Attention: Without repackaging, this method does NOT decrease the HDF5 file's size! 
        See :func:`cait.DataHandler.repackage` for details.

        :param type: The group in the HDF5 set from which the events are deleted,
            typically 'events', 'testpulses' or noise.
        :type type: string
        :param repackage: If set to True, the HDF5 file will be repackaged.
        :type repackage: bool

        >>> dh.drop_raw_data()
        Dataset Event deleted from group events.
        """
        self.drop(group=type, dataset='event', repackage=repackage)

    def downsample_raw_data(self, type: str = 'events', down: int = 16, dtype: str = 'float32',
                            name_appendix: str = '', delete_old: bool = True, batch_size: int = 1000,
                            repackage: bool = False):
        """
        Downsample the dataset "event" from a specified group in the HDF5 file.

        For large scale analysis and limited server space, the converted HDF5 datasets exceed storage space capacities.
        For this scenario, the raw data events can be downsampled by a given factor. Downsampling to sample frequencies
        below 1kHz is in many situations sufficient for viewing events and most features calculations.

        Attention: Without repackaging, this method does NOT decrease the HDF5 file's size! 
        See :func:`cait.DataHandler.repackage` for details.

        :param type: The group in the HDF5 set from which the events are downsampled,
            typically 'events', 'testpulses' or noise.
        :type type: string
        :param down: The factor by which the data is downsampled. This should be a factor of 2.
        :type down: int
        :param dtype: The data type of the new stored events, typically you want this to be float32.
        :type dtype: string
        :param name_appendix: An appendix to the dataset event in order to keep the old and the new events.
        :type name_appendix: string
        :param delete_old: If true, the old events are deleted. Deactivate only, if an unique name_appendix is chosen.
        :type delete_old: bool
        :param batch_size: The batch size for the copy, reduce if you face memory problems.
        :type batch_size: int
        :param repackage: If set to True, the HDF5 file will be repackaged.
        :type repackage: bool

        >>> dh.downsample_raw_data()
        Old Dataset Event deleted from group events.
        New Dataset Event with downsample rate 16 created in group events.
        """

        if name_appendix == '' and not delete_old:
            raise KeyError('To keep the old events please choose appropriate name appendix for the new!')

        with h5py.File(self.get_filepath(), 'r+') as h5f:
            if "event" in h5f[type]:

                h5f[type]['event_temp_'] = h5f[type]['event']
                if delete_old or name_appendix == '':
                    del h5f[type]['event']
                    print(f'Old dataset {fmt_ds("event")} deleted from group {fmt_gr(type)}.')

                # get number batches and shape
                nmbr_channels, nmbr_events, record_length = h5f[type]['event_temp_'].shape
                nmbr_batches = int(nmbr_events / batch_size)

                h5f[type].create_dataset('event' + name_appendix,
                                         shape=(nmbr_channels, nmbr_events, int(record_length / down)),
                                         dtype=dtype)
                print(f'New Dataset {fmt_ds("event"+ name_appendix)} with downsample rate {down} created in group {fmt_gr(type)}.')

                # define function to downsample and write to new data set

                for i in range(nmbr_batches + 1):
                    events = np.array(
                        h5f[type]['event_temp_'][:, (i) * batch_size:(i + 1) * batch_size])
                    if events.shape[1] > 0:
                        events = np.mean(
                            events.reshape((events.shape[0], events.shape[1], int(events.shape[2] / down), down)),
                            axis=3)
                        h5f[type]['event' + name_appendix][:,
                        (i) * batch_size:(i + 1) * batch_size] = events

                print('Done.')

                del h5f[type]['event_temp_']

            else:
                raise FileNotFoundError('There is no event dataset in group {} in the HDF5 file.'.format(type))
            
        if repackage: self.repackage()

    def truncate_raw_data(self, type: str,
                          truncated_idx_low: int,
                          truncated_idx_up: int,
                          dtype: str = 'float32',
                          name_appendix: str = '',
                          delete_old: bool = True,
                          repackage: bool = False):
        """
        Truncate the record window of the dataset "event" from a specified group in the HDF5 file.

        For measurements with high event rate (above ground, ...) a long record window might be counter productive,
        due to more piled up events in the window. For this reason, you can truncate the length of the record window
        with this function.

        Attention: Without repackaging, this method does NOT decrease the HDF5 file's size! 
        See :func:`cait.DataHandler.repackage` for details.

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
        :param repackage: If set to True, the HDF5 file will be repackaged.
        :type repackage: bool
        """

        if name_appendix == '' and not delete_old:
            raise KeyError('To keep the old events please choose appropriate name appendix for the new!')

        with h5py.File(self.get_filepath(), 'r+') as h5f:
            if "event" in h5f[type]:
                events = np.array(h5f[type]['event'])
                if delete_old:
                    del h5f[type]['event']
                    print(f'Old Dataset {fmt_ds("event")} deleted from group {fmt_gr(type)}.')
                events = events[:, :, truncated_idx_low:truncated_idx_up]
                h5f[type].create_dataset('event' + name_appendix, data=events, dtype=dtype)
                print(f'New Dataset {fmt_ds("event"+ name_appendix)} truncated to interval {truncated_idx_low}:{truncated_idx_up} created in group {fmt_gr(type)}.')
            else:
                raise FileNotFoundError('There is no event dataset in group {} in the HDF5 file.'.format(type))
            
        if repackage: self.repackage()

    def get(self, group: str, dataset: str,
            idx0: Union[int, List[Union[int, bool]]] = None, 
            idx1: Union[int, List[Union[int, bool]]] = None, 
            idx2: Union[int, List[Union[int, bool]]] = None):
        """
        Get a dataset from the HDF5 file with save closing of the file stream.
        The additional indices idx0, idx1 and idx2 can be integers, lists of integers or boolean arrays, and are used where appropriate.
        E.g. a 3-dimensional dataset will accept all three indices while a 2d set ignores the last one, etc. 
        If boolean arrays are used, their shape has to match the data's shape along the respective dimension.

        :param group: The name of the group in the HDF5 set.
        :type group: string
        :param dataset: The name of the dataset in the HDF5 set. There are special key word for calculated properties
            from the main parameters, namely 'pulse_height', 'onset', 'rise_time', 'decay_time', 'slope'. These are
            consistent with used in the cut when generating a standard event.
        :type dataset: string
        :param idx0: An index passed to the data set inside the HDF5 file as the first index,
            before it is converted to a numpy array. If left at None value, the slice : operator is passed instead.
        :type idx0: int
        :param idx1: An index passed to the data set inside the HDF5 file as the second index,
            before it is converted to a numpy array. If left at None value, the slice : operator is passed instead.
        :type idx1: int
        :param idx2: An index passed to the data set inside the HDF5 file as the third index,
            before it is converted to a numpy array. If left at None value, the slice : operator is passed instead.
        :type idx2: int
        :return: The dataset from the HDF5 file
        :rtype: numpy array
        """

        # if requested dataset is a virtual dataset, we have to make sure that the original files still 
        # exist. Otherwise the returned data is nonsensical
        with self.get_filehandle(mode="r") as f:
            if dataset in MAINPAR:
                if dataset == "pulse_height" and 'pulse_height' in f[group]:
                    available = ds_source_available(f, group, "pulse_height")
                else: 
                    available = ds_source_available(f, group, "mainpar")
            elif dataset in ADD_MAINPAR:
                available = ds_source_available(f, group, "add_mainpar")
            else:
                available = ds_source_available(f, group, dataset)

        if not available:
            raise FileNotFoundError(f"One or more of the source files for the virtual dataset '{dataset}' in group '{group}' are unavailable.")
        
        # For indices not specified we use all entries along the corresponding axis (equivalent to numpy's [:] operator)
        if idx0 is None: idx0 = slice(None)
        if idx1 is None: idx1 = slice(None)
        if idx2 is None: idx2 = slice(None)

        with self.get_filehandle(mode="r") as f:
            if dataset == 'pulse_height' and 'pulse_height' not in f[group]:
                data = np.array(f[group]['mainpar'][idx0, idx1, 0])
            elif dataset == 'onset':
                data = np.array((f[group]['mainpar'][idx0, idx1, 1] - self.record_length / 4) / self.sample_frequency * 1000)
            elif dataset == 'rise_time':
                data = np.array(
                    (f[group]['mainpar'][idx0, idx1, 2] - f[group]['mainpar'][idx0, idx1, 1]) / self.sample_frequency * 1000)
            elif dataset == 'decay_time':
                data = np.array(
                    (f[group]['mainpar'][idx0, idx1, 6] - f[group]['mainpar'][idx0, idx1, 4]) / self.sample_frequency * 1000)
            elif dataset == 'slope':
                data = np.array(f[group]['mainpar'][idx0, idx1, 8] * self.record_length)
            else:
                for i, name in enumerate(ADD_MAINPAR):
                    if dataset == name and name not in f[group]:
                        data = np.array(f[group]['add_mainpar'][idx0, idx1, i])
                        break
                else:
                    data = f[group][dataset]
                    dim = data.ndim
                    if dim == 3: data = data[idx0, idx1, idx2]
                    elif dim == 2: data = data[idx0, idx1]
                    elif dim == 1: data = data[idx0]
                
                    data = np.array(data)
        return data

    def set(self, 
            group: str, 
            n_channels: int = None, 
            channel: int = None, 
            change_existing: bool = False,
            overwrite_existing: bool = False,
            write_to_virtual: bool = None,
            dtype: str = None,
            **kwargs: List[Union[float, bool]]):
        """
        Include data into the HDF5 file. Datasets are passed as keyword arguments and the keys are used as names for the datasets. 
        E.g. set("events", pulse_heights=data) creates a dataset "pulse_heights" in the group "events". The shape of the dataset matches data's shape.
        Alternatively, one-dimensional data can be written to a multi-dimensional array (as is often necessary for multiple channels).
        This is achieved by specifying the number of desired channels (n_channels) and the channel index (channel) to write to.
        E.g. set("events", n_channels=2, channel=0, pulse_heights=data) creates a "pulse_heights" dataset in the "events" group of shape (2, *data.shape), and `data` is written into the 0-th channel.
        Notice that in most cases you probably want `data` to be of shape (n, ). Otherwise it will probably lead to unexpectedly high-dimensional datasets.

        :param group: The name of the group in the HDF5 file. If it doesn't exist yet, it will be created.
        :type group: string
        :param n_channels: The number of channels that the data should have (first dimension of the dataset).
        :type n_channels: int
        :param channel: The channel that the data gets added to.
        :type channel: int
        :param change_existing: If set to True, already existing datasets are overwritten. For that, the shape and dtype of the new dataset have to match the already existing one's.
        :type change_existing: bool
        :param overwrite_existing: If set to True, already existing datasets are overwritten in case the new dtype and/or shape does not match the existing dtype/shape.
        :type overwrite_existing: bool
        :param write_to_virtual: If set to True and the target dataset is an already existing virtual dataset, the new data is written to the virtual dataset, i.e. it will end up in the source HDF5 files. This might be intended but in most cases, you will probably want this to be set to False to avoid unexpectedly changing remote files. Note that this parameter is None by default and has to be set to True or False when attempting to write to a virtual dataset. Note further, that if set to True, the shape and dtype of the new dataset must match the virtual dataset exactly. Otherwise, a non-remote dataset is created regardless.
        :type write_to_virtual: bool, Default: None
        :param dtype: The desired dtype of the dataset in the HDF5 file. If none is specified, "bool" and "float32" are used for boolean and numeric arrays, respectively.
        :type dtype: string
        :param kwargs: datasets to include (see below)
        :type kwargs: List[Union[float, bool]]

        :Keyword Arguments:
        Pass a keyword argument of the form dataset_name=dataset_data for every dataset that you want to include in the HDF5 file.

        >>> # Include 'data1' and 'data2' as datasets 'new_ds1' and 'new_ds2' in group 'noise' 
        >>> # ('new_ds1' and 'new_ds2' do not yet exist)
        >>> dh.set(group="noise", new_ds1=data1, new_ds2=data2)

        >>> # Include 'data1' and 'data2' as datasets 'ds1' and 'ds2' in group 'noise' 
        >>> # (either or both of 'ds1' and 'ds2' already exist and have correct shape/dtype for new
        >>> # data)
        >>> dh.set(group="noise", ds1=data1, ds2=data2, change_existing=True) 

        >>> # Include 'data1' and 'data2' as datasets 'ds1' and 'ds2' in group 'noise' 
        >>> # (either or both of 'ds1' and 'ds2' already exist and have incorrect shape/dtype for new
        >>> # data, but we want to force the new dtype/shape)
        >>> dh.set(group="noise", ds1=data1, ds2=data2, overwrite_existing=True)

        >>> # Include 'data1' and 'data2' as datasets 'ds1' and 'ds2' in group 'noise' 
        >>> # ('data1' and 'data2' are 1-dimensional but we want to create 2-dimensional 
        >>> # datasets (for different channels e.g.) and write the data into the 0-th channel. This also
        >>> # works for writing single channels to already existing multi-channel datasets.)
        >>> dh.set(group="noise", n_channels=2, channel=0, ds1=data1, ds2=data2)

        >>> # Include 'data1' as dataset 'ds1' in group 'noise' 
        >>> # ('ds1' already exists and is a virtual dataset with matching shape but dtype 'float64'.
        >>> # We want to write to the original data in the respective source files.)
        >>> dh.set(group="noise", ds1=data1, dtype='float64', write_to_virtual=True)

        >>> # Include 'data1' as dataset 'ds1' in group 'noise' 
        >>> # ('ds1' already exists and is a virtual dataset. We want to overwrite it and create a 
        >>> # non-virtual dataset instead)
        >>> dh.set(group="noise", ds1=data1, write_to_virtual=False)
        """

        if np.logical_xor(n_channels==None, channel==None):
            warnings.warn("You have to specify 'n_channels' and 'channel' together", UserWarning)
            return
        
        in_channel_mode = n_channels is not None

        with self.get_filehandle(mode="r+") as f:
            hdf5group = f.require_group(group)
            for key, value in kwargs.items():
                if isinstance(value, list): value = np.array(value)
                if dtype is None: dtype = "bool" if value.dtype == bool else "float32"
                shape = (n_channels, *value.shape) if in_channel_mode else value.shape

                # If a key exists, change_existing has to be set to True.
                # If the new data doesn't match the existing dataset's shape or dtype, it is deleted (if overwrite_existing=True)
                if key in hdf5group.keys():
                    # Check if dataset is virtual. In this case 'write_to_virtual' has to be set.
                    if hdf5group[key].is_virtual and write_to_virtual is None:
                        warnings.warn(f"You are attempting to write to the virtual dataset {fmt_ds(key)}. Set 'write_to_virtual' to True, if you intend to change the original data in the remote file(s), or set it to False to create a new (non-virtual) dataset {fmt_ds(key)} instead.\n", UserWarning)
                        continue

                    # If virtual dataset should not be changed, we delete it and create a new one later
                    # Note that this does NOT delete the dataset from the remote HDF5 files.
                    if hdf5group[key].is_virtual:
                        if write_to_virtual:
                            if (hdf5group[key].shape != shape) or (hdf5group[key].dtype != dtype):
                                warnings.warn(f"The virtual dataset {fmt_ds(key)} has different shape and/or dtype. If you want to write to the original files, these two have to match exactly. Alternatively, you can set 'write_to_virtual' to False to create a new (non-virtual) dataset {fmt_ds(key)}.\nOriginal shape: {hdf5group[key].shape}, New shape: {shape}, Original dtype: {hdf5group[key].dtype}, New dtype: {dtype}\n", UserWarning)
                                continue
                        else:
                            del hdf5group[key]

                    # If there is a shape/dtype mismatch (applies both to virtual and regular datasets), 
                    # the dataset has to be deleted regardless of what else is set
                    elif (hdf5group[key].shape != shape) or (hdf5group[key].dtype != dtype):
                        if overwrite_existing:
                            del hdf5group[key]
                        else:
                            warnings.warn(f"Dataset {fmt_ds(key)} already exists in group {fmt_gr(group)} and has different shape and/or dtype. To overwrite the old set, set 'overwrite_existing' to True. If there is just a dtype mismatch you can also set the dtype using the respective argument.\nOriginal shape: {hdf5group[key].shape}, New shape: {shape}, Original dtype: {hdf5group[key].dtype}, New dtype: {dtype}\n", UserWarning)
                            continue    
                    else:
                        # If dtype and shape agree, change must be explicitly permitted regardless.
                        if not (change_existing or overwrite_existing):
                            warnings.warn(f"Dataset {fmt_ds(key)} already exists in group {fmt_gr(group)}. To change it, set 'change_existing' or 'overwrite_existing' to True\n", UserWarning)
                            continue

                ds = hdf5group.require_dataset(name=key, shape=shape, dtype=dtype)
                if in_channel_mode: 
                    ds[channel, ...] = value
                else: 
                    ds[...] = value

                print(f"Successfully written {fmt_ds(key)} with shape {ds.shape} and dtype '{ds.dtype}' to group {fmt_gr(group)}.\n")

    def rename(self, group: str = None, **kwargs: str):
        """
        Rename groups or datasets in the HDF5 file. Names to change are passed as keyword arguments.

        By default, `group` is set to None. In this case, **kwargs are interpreted as HDF5 group names to change. 
        
        If `group` is set (e.g. to 'events' or 'noise'), **kwargs are interpreted as HDF5 dataset names within that group.

        Notice that we forbid to rename virtual datasets or groups that contain virtual datasets as this could lead to confusion (it is best practice to keep the dataset names between the 'master file' and the source files consistent)

        :param group: The group within which we want to rename datasets. If set to None, groups themselves will be renamed.
        :type group: str, Default: None
        :param kwargs: groups/datasets to be renamed
        :type kwargs: str

        :Keyword Arguments:
        Pass a keyword argument of the form old_name=new_name for every dataset/group that you want to rename in the HDF5 file.

        >>> # Rename groups 'old_group1' and 'old_group2' to 'new_group1' and 'new_group2'
        >>> dh.rename(old_group1='new_group1', old_group2='new_group2')

        >>> # Rename datasets 'old_ds1' and 'old_ds2' in group 'noise' to 'new_ds1' and 'new_ds2'
        >>> dh.rename(group='noise', old_ds1='new_ds1', old_ds2='new_ds2')
        """
        with self.get_filehandle(mode="r+") as h5f:
            if group is None:
                for key, value in kwargs.items():
                    if any([h5f[key][ds].is_virtual for ds in h5f[key].keys()]):
                        print(f"Cannot rename group {fmt_gr(key)} because it contains virtual datasets.")
                        continue

                    h5f.move(key, value)
                    print(f"Successfully renamed group {fmt_gr(key)} -> {fmt_gr(value)}.")
            else:
                for key, value in kwargs.items(): 
                    if h5f[group][key].is_virtual:
                        print(f"Cannot rename virtual dataset {fmt_gr(group)}/{fmt_ds(key)}.")
                        continue

                    h5f[group].move(key, value)
                    print(f"Successfully renamed dataset {fmt_gr(group)}/{fmt_ds(key)} -> {fmt_gr(group)}/{fmt_ds(value)}.")
    
    def keys(self, group: str = None):
        """
        Print the keys of the HDF5 file or a group within it.

        :param group: The name of a group in the HDF5 file of that we print the keys.
        :type group: string or None
        """
        with h5py.File(self.get_filepath(), 'r+') as f:
            if group is None:
                print(list(f.keys()))
            else:
                print(list(f[group].keys()))

    def content(self, group: str = None):
        """
        Print the whole content of the HDF5 and all derived properties. The shape of the datasets as well as their datatypes are also given.

        :param group: The name of a group in the HDF5 file of which we print the content. If None, all groups are printed.
        :type group: string or None
        """

        print(f'The HDF5 file contains the following {fmt_gr("groups")} and {fmt_ds("datasets")}, which can be accessed through get(group, dataset). If present, some contents of the mainpar and add_mainpar datasets are displayed as well. For convenience, they can also be accessed through get(), even though they are not separate datasets in the HDF5 file.\nDatasets marked with {fmt_virt("(v)")} are virtual datasets, i.e. they are stored in another (or multiple other) HDF5 file(s). They are treated like regular datasets but be aware that writing to such datasets actually writes to the respective original files.\n')

        with self.get_filehandle(mode="r") as f:
            if group is None:
                groups = f.keys()
            else:
                # would be nice to have regex support at some point
                groups = [group]

            for group in groups:
                print(fmt_gr(group))
                # move on to next group in case there are no datasets in this group
                if not f[group].keys(): continue

                # 22 is the length of the longest add_mainpar string
                width_dataset = max(len(max(f[group].keys(), key=len)), 22)
                width_shape = max([len(str(f[group][ds].shape)) for ds in f[group].keys()])

                for dataset in f[group].keys():
                    # if dataset is virtual, we include an identifier
                    virt_str = " (v)" if f[group][dataset].is_virtual else ' '*4 

                    print(f'  {fmt_ds(f"{dataset:<{width_dataset+1}}")}{fmt_virt(virt_str)} {str(f[group][dataset].shape):<{width_shape+1}} {f[group][dataset].dtype}')

                    if dataset=='mainpar':
                        shape = f[group]['mainpar'].shape[:2]
                        for dataset in MAINPAR:
                            print(f'  |{dataset:<{width_dataset+len(virt_str)}} {shape}')
                
                    if dataset=='add_mainpar':
                        shape = f[group]['add_mainpar'].shape[:2]
                        for dataset in ADD_MAINPAR:
                            print(f'  |{dataset:<{width_dataset+len(virt_str)}} {shape}')

    def generate_startstop(self):
        """
        Generate a startstop data set in the metainfo group from the testpulses time stamps.
        """

        print('Generating Start Stop Metainfo.')

        with h5py.File(self.get_filepath(), 'r+') as f:

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
        Initialize an empty HDF5 set with name/path as set by dh.set_filepath().
        """
        assert hasattr(self, "path_h5"), "To initialize an empty HDF5 file you have to first set its name/path using dh.set_filepath()."
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
        t = (np.arange(self.record_length) - self.record_length / 4) / self.sample_frequency
        if ms: t *= 1000
        return t

    def repackage(self):
        """
        Repackage the HDF5 file of DataHandler to reduce its file size in case datasets were deleted previously. 

        For large scale analysis and limited server space, the converted HDF5 datasets exceed storage space capacities.
        For this scenario, the raw data events can be deleted after the calculation of all useful features. At a later point, the events can be included again if needed. Similarly, one might have included some temporary datasets which one wishes to delete at a later point to avoid clutter.
        Unwanted datasets can be dropped using :func:`cait.DataHandler.drop` and :func:`cait.DataHandler.drop_raw_data`, HOWEVER this does not reduce the HDF5 file's size due to the tree structure of the HDF5 file!
        For reducing the file size, the HDF5 file has to be repacked with the h5repack method of the HDF5 Tools, see https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack. 
    
        This method is equivalent to and can also be done on Ubuntu/Mac e.g. with

        >>> h5repack test_data/test_001.h5 test_data/test_001_copy.h5
        >>> rm test_data/test_001.h5
        >>> mv test_data/test_001_copy.h5 test_data/test_001.h5
        """

        oldFile = self.get_filepath()
        oldSize = os.path.getsize(oldFile)
        newFile = list(os.path.splitext(oldFile))
        newFile[0] = newFile[0] + '_copy'
        newFile = ''.join(newFile)

        subprocess.run(['h5repack', oldFile, newFile], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        memorySaved = oldSize - os.path.getsize(newFile)
        
        os.replace(newFile, oldFile)
        print(f"Successfully repackaged '{oldFile}'. Memory saved: {sizeof_fmt(memorySaved)}")
