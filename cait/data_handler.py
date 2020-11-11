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


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler(SimulateMixin,
                  RdtMixin,
                  PlotMixin,
                  FeaturesMixin,
                  AnalysisMixin):
    """
    A class for the processing of raw data set. Uses Mixins for most features.
    """

    def __init__(self, run, module, channels, record_length, sample_frequency=25000):
        """
        Give general information of the detector

        :param run: int, the number of the run
        :param module: string, the naming of the detector module
        :param channels: list, the channels in the bck file that belong to the module
        :param record_length: int, the number of samples in one record window
        :param sample_frequency: int, the sample requency of the recording
        """
        # ask user things like which detector working on etc
        if len(channels) != 2:
            raise NotImplementedError('Only for 2 channels implemented.')
        self.run = run
        self.module = module
        self.record_length = record_length
        self.nmbr_channels = len(channels)
        self.channels = channels
        self.sample_frequency = sample_frequency
        self.sample_length = 1000 / self.sample_frequency
        self.t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * self.sample_length

        if self.nmbr_channels == 2:
            self.channel_names = ['Phonon', 'Light']
            self.colors = ['red', 'blue']
        elif self.nmbr_channels == 3:
            self.channel_names = ['Channel 1', 'Channel 2', 'Channel 3']
            self.colors = ['red', 'red', 'blue']

        print('DataHandler Instance created.')

    def set_filepath(self,
                     path_h5: str,
                     fname: str,
                     appendix=True):
        """
        Set the path to the bck_XXX.hdf5 file for further processing.

        :param path_h5: String to directory that contains the runXY_Module folders
        :param fname: String, usually something like bck_xxx
        :return: -
        """

        if appendix:
            if self.nmbr_channels == 2:
                app = '-P_Ch{}-L_Ch{}'.format(*self.channels)
            elif self.nmbr_channels == 3:
                app = '-1_Ch{}-2_Ch{}-3_Ch{}'.format(*self.channels)
        else:
            app = ''

        # check if the channel number matches the file, otherwise error
        self.path_h5 = "{}/run{}_{}/{}{}.h5".format(path_h5, self.run, self.module,
                                                                fname, app)
        self.fname = fname


    def import_labels(self,
                      path_labels,
                      type='events',
                      path_h5=None):
        """
        Include the *.csv file with the labels into the HDF5 File.
        :param path_labels: string, path to the folder that contains the run_module folder
            e.g. "data" --> look for labels in "data/runXY_moduleZ/labels_bck_0XX_<type>"
        :param type: string, the type of labels, typically "events" or "testpulses"
        :param path_h5: string, optional, provide an extra (full) path to the hdf5 file
            e.g. "data/hdf5s/bck_001[...].h5"
        :return: -
        """

        if not path_h5:
            path_h5 = self.path_h5

        path_labels = '{}/run{}_{}/labels_{}_{}.csv'.format(
            path_labels, self.run, self.module, self.fname, type)

        h5f = h5py.File(path_h5, 'r+')

        if path_labels != '' and os.path.isfile(path_labels):
            labels = np.genfromtxt(path_labels)
            labels = labels.astype('int32')
            length = len(labels)
            labels.resize((2, int(length / 2)))

            print(h5f.keys())

            events = h5f[type]

            if "labels" in events:
                events['labels'][...] = labels
                print('Edited Labels.')

            else:
                events.create_dataset('labels', data=labels)
                events['labels'].attrs.create(name='unlabeled', data=0)
                events['labels'].attrs.create(name='Event_Pulse', data=1)
                events['labels'].attrs.create(
                    name='Test/Control_Pulse', data=2)
                events['labels'].attrs.create(name='Noise', data=3)
                events['labels'].attrs.create(name='Squid_Jump', data=4)
                events['labels'].attrs.create(name='Spike', data=5)
                events['labels'].attrs.create(
                    name='Early_or_late_Trigger', data=6)
                events['labels'].attrs.create(name='Pile_Up', data=7)
                events['labels'].attrs.create(name='Carrier_Event', data=8)
                events['labels'].attrs.create(
                    name='Strongly_Saturated_Event_Pulse', data=9)
                events['labels'].attrs.create(
                    name='Strongly_Saturated_Test/Control_Pulse', data=10)
                events['labels'].attrs.create(
                    name='Decaying_Baseline', data=11)
                events['labels'].attrs.create(name='Temperature Rise', data=12)
                events['labels'].attrs.create(name='Stick Event', data=13)
                events['labels'].attrs.create(name='Sawtooth Cycle', data=14)
                events['labels'].attrs.create(name='unknown/other', data=99)

                print('Added Labels.')

        elif (path_labels != ''):
            print("File '{}' does not exist.".format(path_labels))

    def import_predictions(self,
                           model,
                           path_predictions,
                           type='events',
                           path_h5=None):
        """
        Include the *.csv file with the predictions of a model into the HDF5 File.
        :param model: string, the naming for the type of model, e.g. Random Forest --> "RF"
        :param path_predictions: string, path to the folder that contains the csv file
            e.g. "data" --> look for predictions in "data/<model>_predictions_<self.fname>_<type>"
        :param type: string, the type of labels, typically "events" or "testpulses"
        :param path_h5: string, optional, provide an extra (full) path to the hdf5 file
            e.g. "data/hdf5s/bck_001[...].h5"
        :return: -
        """

        if not path_h5:
            path_h5 = self.path_h5

        path_predictions = '{}{}_predictions_{}_{}.csv'.format(
            path_predictions, model, self.run, self.module, self.fname, type)

        h5f = h5py.File(path_h5, 'r+')

        if path_predictions != '' and os.path.isfile(path_predictions):
            labels = np.genfromtxt(path_predictions)
            labels = labels.astype('int32')
            length = len(labels)
            labels.resize((2, int(length / 2)))

            print(h5f.keys())

            events = h5f[type]

            if "{}_predictions".format(model) in events:
                events["{}_predictions".format(model)][...] = labels
                print('Edited Predictions.')

            else:
                events.create_dataset("{}_predictions".format(model), data=labels)
                events["{}_predictions".format(model)].attrs.create(name='unlabeled', data=0)
                events["{}_predictions".format(model)].attrs.create(name='Event_Pulse', data=1)
                events["{}_predictions".format(model)].attrs.create(
                    name='Test/Control_Pulse', data=2)
                events["{}_predictions".format(model)].attrs.create(name='Noise', data=3)
                events["{}_predictions".format(model)].attrs.create(name='Squid_Jump', data=4)
                events["{}_predictions".format(model)].attrs.create(name='Spike', data=5)
                events["{}_predictions".format(model)].attrs.create(
                    name='Early_or_late_Trigger', data=6)
                events["{}_predictions".format(model)].attrs.create(name='Pile_Up', data=7)
                events["{}_predictions".format(model)].attrs.create(name='Carrier_Event', data=8)
                events["{}_predictions".format(model)].attrs.create(
                    name='Strongly_Saturated_Event_Pulse', data=9)
                events["{}_predictions".format(model)].attrs.create(
                    name='Strongly_Saturated_Test/Control_Pulse', data=10)
                events["{}_predictions".format(model)].attrs.create(
                    name='Decaying_Baseline', data=11)
                events["{}_predictions".format(model)].attrs.create(name='Temperature Rise', data=12)
                events["{}_predictions".format(model)].attrs.create(name='Stick Event', data=13)
                events["{}_predictions".format(model)].attrs.create(name='Sawtooth Cycle', data=14)
                events["{}_predictions".format(model)].attrs.create(name='unknown/other', data=99)

                print('Added {} Predictions.'.format(model))

    def get_filehandle(self, path=None):
        """
        Returns the opened filestream to the hdf5 file
        :param path: string, optional, give a path to the hdf5 file
        :return: h5py file object
        """
        if path is None:
            f = h5py.File(self.path_h5, 'r+')
        else:
            f = h5py.File(path, 'r+')
        return f
