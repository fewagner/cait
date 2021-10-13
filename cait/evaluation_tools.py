# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import os
import h5py
import numpy as np
import math

import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .evaluation._color import console_colors, mpl_default_colors
from .evaluation._pgf_config import set_mpl_backend_pgf, set_mpl_backend_fontsize
from .styles._plt_styles import use_cait_style, make_grid

from .features._mp import *

import json
from jupyter_dash import JupyterDash as Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class EvaluationTools:
    """
    The Class EvaluationTools provides a easier way to handle the data.
    Especially for hdf5-files as widely used in this library.
    Besides that it also provides a easy way to organize predictions as well
    as visualize the results via confusion matrix and TSNE or PCA plots to depict high dimensional data.
    How it is used is best seen in the tutorial 'Machine Learning-based Event Selection'.
    """

    def __init__(self):
        self.save_plot_dir = './'
        self.save_as = False

        self.event_nbrs = None  # numeration of all events

        self.data = None  # can contain part of mainpar or events
        self.features = None  # normalized 'data'

        self.events = None  # 'events/event' from hdf5 file
        self.mainpar = None  # 'events/mainpar' from hdf5 file
        self.mainpar_labels = None  # 'events/mainpar'.attrs from hdf5 file

        self.pl_channel = None
        self.files = None
        self.file_nbrs = None
        self.events_per_file = None

        self.label_nbrs = None
        self.labels = None
        self.labels_color_map = None

        self.test_size = 0.50
        self.is_train_valid = False
        self.is_train = None
        self.predictions = None
        self.perdiction_true_labels = None

        self.color_order = mpl_default_colors

        self.scaler = StandardScaler()

    # ################### PRIVATE ###################

    def __add_data(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data = np.vstack([self.data, data])

    def __add_events(self, events):
        if self.events is None:
            self.events = events
        else:
            self.events = np.vstack([self.events, events])

    def __add_mainpar(self, mainpar, mainpar_labels):
        if self.mainpar is None:
            self.mainpar = mainpar
        else:
            self.mainpar = np.vstack([self.mainpar, mainpar])

        if self.mainpar_labels is None:
            self.mainpar_labels = dict([(k, v) for k, v in mainpar_labels])
        else:
            for k, v in mainpar_labels:
                if k not in self.mainpar_labels:
                    if v in self.mainpar_labels.values():
                        raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                                         "Every label for the mainparameters must have a unique value.\n" +
                                         "Currently containing labels: {}\n".format(self.mainpar_labels) +
                                         "Tried to add: {}".format(mainpar_labels))
                    self.mainpar_labels[k] = v

    def __add_files(self, file):
        if self.files is None:
            self.files = [file]
            self.events_per_file = [0]
        else:
            if file not in self.files:
                self.files.append(file)
                self.events_per_file.append(0)

    def __add_label_nbrs(self, label_nbrs):
        if self.label_nbrs is None:
            self.label_nbrs = np.array(label_nbrs, dtype=int)
        else:
            self.label_nbrs = np.hstack([self.label_nbrs, np.array(label_nbrs, dtype=int)])
        self.__add_labels_color_map()

    def __add_labels_color_map(self, verb=False):
        self.labels_color_map = {}
        possible_labels = np.unique(self.get_label_nbrs(what='all', verb=verb))
        if self.predictions is not None:
            for pm in self.predictions.keys():
                possible_labels = np.hstack(
                    [possible_labels, np.unique(self.get_pred(pred_method=pm, verb=verb))])

        for i, ln in enumerate(np.unique(possible_labels)):
            # for i, ln in enumerate(np.unique([self.get_label_nbrs(what='all')])):
            self.labels_color_map[ln] = self.color_order[i]

    def __add_labels(self, labels):
        if self.labels is None:
            self.labels = dict([(val, name) for (name, val) in labels])
        else:
            for name, val in labels:
                if (name, val) not in self.labels.keys():
                    self.labels[val] = name

    def __add_event_nbrs(self, event_nbrs):
        if self.event_nbrs is None:
            self.event_nbrs = event_nbrs
        else:
            self.event_nbrs = np.hstack([self.event_nbrs, event_nbrs])

    def __add_file_nbrs(self, file_nbrs):
        if self.file_nbrs is None:
            self.file_nbrs = file_nbrs
        else:
            self.file_nbrs = np.hstack([self.file_nbrs, file_nbrs])

    def __check_train_set(self, verb=False):
        if not self.is_traintest_valid:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "No valid test set available. " +
                      "A test set of size {} is generated.".format(self.test_size))
            self.split_test_train(self.test_size)

    # ################### OTHER ###################

    def add_events_from_file(self,
                             file,
                             channel,
                             which_data='mainpar',
                             all_labeled=False,
                             only_idx=None,
                             force_add=False,
                             verb=False):
        """
        Reads in a labels, data from a channel of a given hdf5 file and
        adds this data to the properties

        :param file: Path to hd5 file from which the data should be read.
        :type file: string
        :param channel: The number of the channel.
        :type channel: int
        :param which_data: Default 'mainpar',select which data should be used as data (e.g. mainparameters, additional mainparameters, timeseries) if set to none then data is keept empty. It is also possible to set this paramater to an array of the length of the labels which are then stored in data.
        :type which_data: string or 2D array
        :param all_labeled: Default False, flag is set, include exactly the events that are labeled.
        :type all_labeled: boolean
        :param only_idx: Indices only include in the dataset then only use these.
        :type only_idx: list of int
        :param force_add: Default False, lets you add a file twice when set to True.
        :type force_add: boolean
        :param verb: Default False, if True additional messages are printed.
        :type verb: boolean
        """

        # --------------------------------------------
        # INPUT HANDLING
        # --------------------------------------------

        # check if the file input has type string
        if type(file) is not str:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "The variable must be of type string and not '{}'.".format(type(file)))

        # check if the file is accessable
        if not os.path.isfile(file):
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "The given hdf5 file '{}' does not exist.".format(file))

        # check if the data key is correct
        if not (type(which_data) == list or type(which_data) == np.ndarray or
                which_data == 'mainpar' or which_data == 'add_mainpar' or which_data == 'timeseries' or
                which_data == None):
            raise ValueError(console_colors.FAIL + "WARNING: " + console_colors.ENDC +
                             "Only 'mainpar', 'add_mainpar', 'timeseries', list of data or None are valid options to read from file.")

        # check if we look at a correct channel
        if type(channel) is int:
            if channel <= 2:
                self.pl_channel = channel
            else:
                raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                                 "Parameter 'pl_channel = {}' has no valid value.\n".format(channel) +
                                 "(Use '0' for the phonon and '1' for the light in a 2-channel detector setup)\n" +
                                 "(Use '0,1' for the phonon and '2' for the light in a 3-channel detector setup)")
        else:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Parameter 'pl_channel' has to be of type int.")

        # if file already exists ask again
        if force_add == True and self.files != None and file in self.files:
            print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                  "Data from this file already exists. Continue? [y/N]")
            cont = input()
            if cont != 'y' and cont != 'Y':
                return

        # --------------------------------------------
        # THE ACTUAL HAPPENING
        # --------------------------------------------

        # here the files are added as data source
        self.__add_files(file)

        with h5py.File(file, 'r') as ds:

            # check if the hdf5 set has events
            if 'events' not in ds.keys():
                raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                                 "No group 'events' in the provided hdf5 file '{}'.".format(file))
            # if the all_labeled flag is set, include exactly the events that are labeled
            use_idx = []
            if all_labeled:
                if only_idx is None:
                    length_events = ds['events/event'].shape[1]
                    only_idx = [i for i in range(length_events)]
                    del length_events
                for i in only_idx:
                    if ds['events/labels'][channel, i] != 0:
                        use_idx.append(i)
                nbr_events_added = len(use_idx)
            # elif we got a list of idx to only include in the dataset then only use these
            elif only_idx is not None:
                nbr_events_added = len(only_idx)
                use_idx = only_idx
            else:
                nbr_events_added = ds['events/event'].shape[1]
                use_idx = [i for i in range(nbr_events_added)]

            # find the index of the current file in the list of files
            file_index = self.files.index(file)

            if (type(which_data) is str) and (which_data == 'mainpar'):
                self.__add_data(
                    np.delete(ds['events/mainpar'][channel, use_idx, :], ds['events/mainpar'].attrs['offset'], axis=1))
            if (type(which_data) is str) and (which_data == 'add_mainpar'):
                self.__add_data(
                    np.copy(ds['events/add_mainpar'][channel, use_idx, :]))
            elif (type(which_data) is str) and (which_data == 'timeseries'):
                self.__add_data(
                    np.copy(ds['events/event'][channel, use_idx, :]))
            # elif which_data is None:
            # self.__add_data(np.array([]))
            # do nothing
            elif type(which_data) == list or type(which_data) == np.ndarray:
                tmp_which_data = np.array(which_data)
                if not (tmp_which_data.shape[0] == np.array(ds['events/labels'][channel, use_idx]).shape[0]):
                    raise ValueError(console_colors.FAIL + "WARNING: " + console_colors.ENDC +
                                     "Given 'which_data' must be of size {} but is of size {}".format(
                                         np.array(ds['events/labels']
                                                  [channel, use_idx]).shape[0],
                                         tmp_which_data.shape[0]))
                self.__add_data(tmp_which_data)

            # add also the events and the mainpar seperately
            self.__add_events(ds['events/event'][channel, use_idx, :])
            self.__add_mainpar(
                ds['events/mainpar'][channel, use_idx, :], ds['events/mainpar'].attrs.items())

            # if there are labels in the h5 set then also add them
            if 'labels' in ds['events']:
                # add the labels of all the events
                self.__add_label_nbrs(
                    np.copy(ds['events/labels'][channel, use_idx]))

                # add the colors that correspond to the labels
                if len(np.unique(self.label_nbrs)) > len(self.color_order):
                    print(console_colors.FAIL + "WARNING: " + console_colors.ENDC +
                          "The color_order only contains {} where as there are {} unique labels.".format(
                              len(self.color_order), len(np.unique(self.label_nbrs))))

                # add the namings of the labels
                self.__add_labels(list(ds['events/labels'].attrs.items()))

            # if there are no labels, add the events all as unlabeled
            else:
                self.__add_label_nbrs([0] * nbr_events_added)
                self.__add_labels([('unlabeled', 0)])
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "In the provided hdf5 file are no label_nbrs available and therefore are set to 'unlabeled'.")

            # add a numbering to each event to which file it belongs
            self.__add_file_nbrs(np.full(nbr_events_added, file_index))

            # add a continuous numbering to the events such that they are uniquely identifiable
            self.__add_event_nbrs(
                np.arange(self.events_per_file[file_index], self.events_per_file[file_index] + nbr_events_added))
            # increase the value of the stored counter of the events per file
            self.events_per_file[file_index] += nbr_events_added
            # reset the train_test split variable because new data was added
            self.is_traintest_valid = False

        if verb:
            print('Added Events from file to instance.')

    def set_data(self, data):
        """
        Replaces mainparameters or timeseries with a chosen data set of data.

        :param data: Dataset which is analysed.
        :type data: array
        """

        if data.shape[0] != self.events.shape[0]:
            raise ValueError(console_colors.FAIL + "WARNING: " + console_colors.ENDC +
                             "The the length of data {} does not ".format(data.shape[0]) +
                             "correspond to the number of events {}.".format(self.events.shape[0]))

        self.data = data
        self.gen_features()

    def set_scaler(self, scaler):
        """
        Sets the scaler for generating the features from the data set.
        Per default the sklearn.preprocessing.StandardScaler() is used.

        :param scaler: Scaler for normalizing the data.
        :type scaler: object
        """
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
        self.gen_features()

    def gen_features(self):
        """
        Normalizes the data and saves it into features.
        """
        if type(self.scaler) == type(StandardScaler()):
            self.scaler.fit(self.data)
            self.features = self.scaler.transform(self.data)
        else:
            print("If the StandardScaler is not used the features have to " +
                  "be transformed manually using 'set_features()'.")

    def set_features(self, features):
        """
        If the StandardScaler is not used features have to be set manually,
        e.g. by using this function.

        :param features: Manual generated features.
        :type features: array
        """
        features = np.array(features)
        if self.data.shape != features.shape:
            raise ValueError(console_colors.FAIL + "WARNING: " + console_colors.ENDC +
                             "The shape of features {}".format(features.shape) +
                             " has to be the same as" +
                             " shape of data {}".format(self.data.shape) +
                             ", since they should just be transformed.")
        self.features = features

    def add_prediction(self, pred_method, pred, true_labels=False, verb=False):
        """
        Adds a new prediction method with labels to the predictions property.

        :param pred_method: The name of the model that made the predictions.
        :type pred_method: string
        :param pred: Contains the predicted labels for the events.
        :type pred: list of int
        :param true_labels: Default False, set to True when predicted labels correspond to actual label numbers (as in superviced learning methods).
        :type true_labels: boolean
        :param verb: Default False, if True addtional output is printed to the console.
        :type verb: boolean
        """
        if len(pred) != len(self.label_nbrs):
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "The parameter 'pred' must have the same amoutn of entries ({}) as ".format(len(pred)) +
                             "events ({}).".format(len(self.label_nbrs)))

        if self.predictions is None:
            self.predictions = dict([(pred_method, (true_labels, pred))])
        else:
            if pred_method not in self.predictions.keys():
                self.predictions[pred_method] = (true_labels, pred)
            elif verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "A prediction method with the name '{}' allready exists.".format(pred_method))

        self.__add_labels_color_map(verb=verb)
        if verb:
            print('Added Predictions to instance.')

    def save_prediction(self, pred_method, path, fname, channel):
        """
        Saves the predictions as a CSV file

        :param pred_method: The name of the model that made the predictions.
        :type pred_method: string
        :param path: Path to the folder that should contain the predictions, e.g. 'predictions/' leads to correct directory.
        :type path: string
        :param fname: The name of the file, e.g. "bck_001".
        :type fname: string
        :param channel: The number of the channel in the module, e.g. Phonon 0, Light 1.
        :type channel: int
        :param verb: Default False, if True  additional ouput is printed.
        :type verb: boolean
        """
        if self.predictions is None:
            raise AttributeError('Add predictions first!')

        if path == '':
            path = './'
        if path[-1] != '/':
            path = path + '/'
        np.savetxt(path + pred_method + '_predictions_' + fname + '_events_Ch' + str(channel) + '.csv',
                   np.array(self.predictions[pred_method][1]),
                   fmt='%i', delimiter='\n')  # the index 1 should access the pred

        print('Saved Predictions as CSV file.')

    def split_test_train(self, test_size, verb=False):
        """
        Seperates the dataset into a training set and a test set with the
        size determined by the input test_size in percent.

        :param test_size: Size of the test set.
        :type test_size: float in (0,1)
        :param verb: Default False, if True additional output is printed.
        :type verb: boolean
        """
        if test_size <= 0 or test_size >= 1:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "The parameter 'test_size' can only have values between 0 and 1.")

        self.test_size = test_size
        total_nbr = np.sum(self.events_per_file)
        event_num = np.arange(total_nbr)
        event_num_train, _ = train_test_split(
            event_num, test_size=test_size)

        self.is_train = np.isin(event_num, event_num_train)
        self.is_traintest_valid = True

        self.gen_features()
        if verb:
            print('Data set splitted.')

    def convert_to_labels(self, label_nbrs, verb=False):
        """
        Converts given label numbers to the corresponding label.

        :param label_nbrs: Contains the label numbers.
        :type label_nbrs: list of int
        :param verb: Default False, if True addtional output is printed to the console.
        :type verb: boolean
        :return: Labels which correspond to the label numbers.
        :rtype: list
        """
        unique_label_nbrs = np.unique(label_nbrs)
        if not np.isin(unique_label_nbrs, list(self.labels.keys())).all():
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "Given label numbers contain unkown labels.")
            return None
        return np.array([self.labels[i] for i in label_nbrs])

    def convert_to_colors(self, label_nbrs, verb=False):
        """
        Converts given label numbers into colors for matplotlib.

        :param label_nbrs: Contains the label numbers.
        :type label_nbrs: list of int
        :param verb: Default False, if True addtional output is printed to the console.
        :type verb: boolean
        :return: Same color for the same labels.
        :rtype: list of colors
        """
        # unique_label_dict = dict(
        #     [(l, i) for i, l in enumerate(np.unique(label_nbrs))])
        # if len(unique_label_dict) > len(self.color_order):
        #     if verb:
        #         print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
        #               "Given label numbers contain more labels than supported colors.")
        #     return None
        # return [self.color_order[unique_label_dict[l]] for l in label_nbrs]
        return [self.labels_color_map[i] for i in label_nbrs]

    def convert_to_labels_colors(self, label_nbrs, return_legend=False, verb=False):
        """
        Converts given label numbers into colors for matplotlib.

        :param label_nbrs: Contain the label numbers.
        :type label_nbrs: list of int
        :param return_legend: Default False, if True a legend in format for matplotlib is returned additionally.
        :type return_legend: boolean
        :param verb: Default False, if True addtional output is printed to the console.
        :type verb: boolean
        :return: List of colors, optional legend for matplotlib.
        :rtype: list of labels
        """
        labels = self.convert_to_labels(label_nbrs, verb=verb)
        colors = self.convert_to_colors(label_nbrs, verb=verb)

        if return_legend:
            legend = np.unique([labels, label_nbrs, colors], axis=1)
            return labels, colors, legend

        return labels, colors

    # ################### GETTER ###################

    def get_train(self, verb=False):
        """
        Getter-function which returns the data from the training set
        in the following order:
        1. event_nbrs
        2. data
        3. features
        4. file_nbrs
        5. label_nbrs

        :param verb: enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: tuple of size 6, where every entry in this tuple is an array
        :rtype: tuple
        """
        self.__check_train_set(verb=verb)
        return self.__get_train_event_nbrs(verb=verb), \
               self.__get_train_data(verb=verb), \
               self.__get_train_features(verb=verb), \
               self.__get_train_file_nbrs(verb=verb), \
               self.__get_train_label_nbrs(verb=verb)

    def get_test(self, verb=False):
        """
        Getter-function which returns the data from the test set
        in the following order:
        1. event_nbrs
        2. data
        3. features
        4. file_nbrs
        5. label_nbrs

        :param verb: enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: tuple of size 6, where every entry in this tuple is an array
        :rtype: tuple
        """

        self.__check_train_set(verb=verb)
        return self.__get_test_event_nbrs(verb=verb), \
               self.__get_test_data(verb=verb), \
               self.__get_test_features(verb=verb), \
               self.__get_test_file_nbrs(verb=verb), \
               self.__get_test_label_nbrs(verb=verb)

    # ------- get event numbers -------
    def __get_train_event_nbrs(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.event_nbrs[self.is_train]

    def __get_test_event_nbrs(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.event_nbrs[np.logical_not(self.is_train)]

    def get_event_nbrs(self, what='all', verb=False):
        """
        Getter-function which returns the event_nbrs

        :param what: Defines what should be returned, which is either 'all' for all event_nbrs, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of event_nbrs depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_event_nbrs(verb=verb)
        elif what == 'test':
            return self.__get_test_event_nbrs(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.event_nbrs

    # ------- get data -------
    def __get_train_data(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.data[self.is_train, :]

    def __get_test_data(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.data[np.logical_not(self.is_train)]

    def get_data(self, what='all', verb=False):
        """
        Getter-function which returns the data

        :param what: Defines what should be returned, which is either 'all' for all data, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of data depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_data(verb=verb)
        elif what == 'test':
            return self.__get_test_data(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.data

    # ------- get mainparameters -------
    def __get_train_mainpar(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.mainpar[self.is_train, :]

    def __get_test_mainpar(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.mainpar[np.logical_not(self.is_train)]

    def get_mainpar(self, what='all', verb=False):
        """
        Getter-function which returns the mainpar

        :param what: Defines what should be returned, which is either 'all' for all mainpar, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of mainpar depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_mainpar(verb=verb)
        elif what == 'test':
            return self.__get_test_mainpar(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.mainpar

    # ------- get events -------
    def __get_train_events(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.events[self.is_train, :]

    def __get_test_events(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.events[np.logical_not(self.is_train)]

    def get_events(self, what='all', verb=False):
        """
        Getter-function which returns the events

        :param what: Defines what should be returned, which is either 'all' for all events, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of events depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_events(verb=verb)
        elif what == 'test':
            return self.__get_test_events(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.events

    # ------- get features (normalized data) -------
    def __get_train_features(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.features[self.is_train]

    def __get_test_features(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.features[np.logical_not(self.is_train)]

    def get_features(self, what='all', verb=False):
        """
        Getter-function which returns the features

        :param what: Defines what should be returned, which is either 'all' for all features, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of features depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_features(verb=verb)
        elif what == 'test':
            return self.__get_test_features(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.features

    # ------- get file nbrs -------
    def __get_train_file_nbrs(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.file_nbrs[self.is_train]

    def __get_test_file_nbrs(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.file_nbrs[np.logical_not(self.is_train)]

    def get_file_nbrs(self, what='all', verb=False):
        """
        Getter-function which returns the file_nbrs

        :param what: Defines what should be returned, which is either 'all' for all file_nbrs, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of file_nbrs depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_file_nbrs(verb=verb)
        elif what == 'test':
            return self.__get_test_file_nbrs(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.file_nbrs

    # ------- get file nbrs -------
    def __get_train_label_nbrs(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.label_nbrs[self.is_train]

    def __get_test_label_nbrs(self, verb=False):
        self.__check_train_set(verb=verb)
        return self.label_nbrs[np.logical_not(self.is_train)]

    def get_label_nbrs(self, what='all', verb=False):
        """
        Getter-function which returns the label_nbrs

        :param what: Defines what should be returned, which is either 'all' for all label_nbrs, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of label_nbrs depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_label_nbrs(verb=verb)
        elif what == 'test':
            return self.__get_test_label_nbrs(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return self.label_nbrs

    def __get_train_labels_in_color(self, verb=False):
        return [self.labels_color_map[i] for i in self.__get_train_label_nbrs(verb=verb)]

    def __get_test_labels_in_color(self, verb=False):
        return [self.labels_color_map[i] for i in self.__get_test_label_nbrs(verb=verb)]

    def get_labels_in_color(self, what='all', verb=False):
        """
        Getter-function which returns the labels_in_color, which can be usefull for plotting.

        :param what: Defines what should be returned, which is either 'all' for all labels_in_color, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of labels_in_color depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_labels_in_color(verb=verb)
        elif what == 'test':
            return self.__get_test_labels_in_color(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return [self.labels_color_map[i] for i in self.get_label_nbrs()]

    # ------- get prediction (predicted label nbrs) -------
    def __get_train_pred(self, pred_method, verb=False):
        self.__check_train_set(verb=verb)
        if pred_method not in list(self.predictions.keys()):
            if verb:
                print(console_colors.WARNING + "NOTE: " + console_colors.ENDC +
                      "The prediction method '{}' does not exist in the stored predictions.".format(pred_method))
            return False
        return self.predictions[pred_method][1][self.is_train]

    def __get_test_pred(self, pred_method, verb=False):
        self.__check_train_set(verb=verb)
        if pred_method not in list(self.predictions.keys()):
            if verb:
                print(console_colors.WARNING + "NOTE: " + console_colors.ENDC +
                      "The prediction method '{}' does not exist in the stored predictions.".format(pred_method))
            return False
        return self.predictions[pred_method][1][np.logical_not(self.is_train)]

    def get_pred(self, pred_method, what='all', verb=False):
        """
        Getter-function which returns the prediction of a method.
        The prediction has to be added at first and can be selected by providing the same abbreviation while adding it.
        The selection is done via the pred_method parameter.

        :param pred_method: Parameter to select the prediction from a certain prediction method, which must be the same string as when added.
        :type pred_method: str
        :param what: Defines what should be returned, which is either 'all' for all prediction, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of prediction depending on the chosen prediction methode (pred_meth) and on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_pred(pred_method, verb=verb)
        elif what == 'test':
            return self.__get_test_pred(pred_method, verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            if pred_method not in list(self.predictions.keys()):
                if verb:
                    print(console_colors.WARNING + "NOTE: " + console_colors.ENDC +
                          "The prediction method '{}' does not exist in the stored predictions.".format(pred_method))
                return False
            return self.predictions[pred_method][1]

    def __get_train_pred_in_color(self, pred_method, verb=False):
        self.__check_train_set(verb=verb)
        if pred_method not in list(self.predictions.keys()):
            if verb:
                print(console_colors.WARNING + "NOTE: " + console_colors.ENDC +
                      "The prediction method '{}' does not exist in the stored predictions.".format(pred_method))
            return False
        if self.get_pred_true_labels(pred_method):
            return [self.labels_color_map[i] for i in self.__get_train_pred(pred_method, verb=verb)]
        else:
            return [self.color_order[i] for i in self.__get_train_pred(pred_method, verb=verb)]

    def __get_test_pred_in_color(self, pred_method, verb=False):
        self.__check_train_set(verb=verb)
        if pred_method not in list(self.predictions.keys()):
            if verb:
                print(console_colors.WARNING + "NOTE: " + console_colors.ENDC +
                      "The prediction method '{}' does not exist in the stored predictions.".format(pred_method))
            return False
        if self.get_pred_true_labels(pred_method):
            return [self.labels_color_map[i] for i in self.__get_test_pred(pred_method, verb=verb)]
        else:
            return [self.color_order[i] for i in self.__get_test_pred(pred_method, verb=verb)]

    def get_pred_in_color(self, pred_method, what='all', verb=False):
        """
        Getter-function which returns the color coded prediction of a method.
        The prediction has to be added at first and can be selected by providing the same abbreviation while adding it.
        The selection is done via the pred_method parameter.

        :param pred_method: Parameter to select the prediction from a certain prediction method get color coded, which must be the same string as when added.
        :type pred_method: str
        :param what: Defines what should be returned, which is either 'all' for all color coded predicitons, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of the color coded predictions depending on the chosen prediction methode (pred_meth) and on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_pred_in_color(pred_method, verb=verb)
        elif what == 'test':
            return self.__get_test_pred_in_color(pred_method, verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            if pred_method not in list(self.predictions.keys()):
                if verb:
                    print(console_colors.WARNING + "NOTE: " + console_colors.ENDC +
                          "The prediction method '{}' does not exist in the stored predictions.".format(pred_method))
                return False

            if self.get_pred_true_labels(pred_method):
                return [self.labels_color_map[i] for i in self.get_pred(pred_method, verb=verb)]
            else:
                return [self.color_order[i] for i in self.get_pred(pred_method, verb=verb)]

    # ------- get if prediction labels correspond to the actual labels -------
    def get_pred_true_labels(self, pred_method):
        """
        Returns if the labels in the prediction correspond to the actual labels, as in the dict self.labels.

        :param pred_method: Abbreviation of the chosen prediction method
        :type pred_method: str
        :return: Return True or False if the labels of the chosen prediction method correspond to the actual labels
        :rtype: bool
        """
        return self.predictions[pred_method][0]

    # ------- get filepaths -------
    def __get_train_filepaths(self, verb=False):
        self.__check_train_set(verb=verb)
        return [self.files[f] for f in self.file_nbrs[self.is_train]]

    def __get_test_filepaths(self, verb=False):
        self.__check_train_set(verb=verb)
        return [self.files[f] for f in self.file_nbrs[np.logical_not(self.is_train)]]

    def get_filepaths(self, what='all', verb=False):
        """
        Getter-function which returns the filepaths for every event.

        :param what: Defines what should be returned, which is either 'all' for all filepaths, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        :return: Returns the part of labels_in_color depending on the parameter what
        :rtype: array
        """
        if what == 'train':
            return self.__get_train_filepaths(verb=verb)
        elif what == 'test':
            return self.__get_test_file_nbrs(verb=verb)
        else:
            if verb and what != 'all':
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")
            return [self.files[f] for f in self.file_nbrs]

    # ################### PLOT ###################

    def plot_event(self, index, what='all', plot_mainpar=False, text=None, verb=False):
        """
        Plots a single event from an given index..

        :param index: The event index which should be plotted in respect to the what parameter
        :type index: int
        :param what: Defines what should be returned, which is either 'all' for all filepaths, 'test' for the test set and 'train' for the trainings set , defaults to 'all'
        :type what: str, optional
        :param plot_mainpar: If True, it adds main parameters to the plot, default False
        :type plot_mainpar: str, optional
        :param text: Adds text to the plot, default None
        :type text: str, optional
        :param verb: Enables addiational output which can be usefull for debugging, defaults to False
        :type verb: bool, optional
        """
        if verb:
            print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                  "index='{}'.".format(index))

        if what not in ['all', 'test', 'train']:
            what = 'all'
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")

        use_cait_style()

        event = self.get_events(what)[index]
        if plot_mainpar:
            event_mainpar = calc_main_parameters(event)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # import ipdb; ipdb.set_trace()
        ax.plot(event, linewidth=1, zorder=-1)
        if plot_mainpar:
            event_mainpar.plotParameters(fig=fig)

        ax.set_ylim(ax.get_ylim())

        if plot_mainpar:
            ax.hlines(y=event_mainpar.offset, xmin=ax.get_xlim()[
                0], xmax=ax.get_xlim()[1], linestyles='--', alpha=0.2)
        ax.vlines(x=int(event.shape[0] / 4), ymin=ax.get_ylim()
        [0], ymax=ax.get_ylim()[1], linestyles='--', alpha=0.2)

        if text is not None:
            ax.set_title(text)

        fig.tight_layout()

        fig.show()

    def plt_pred_with_tsne(self, pred_methods, what='all', plt_labels=True,
                           figsize=None, perplexity=30, as_cols=False, rdseed=None,
                           dot_size=5, verb=False):
        """
        Plots data with TSNE when given a one or a list of predictions method
        to compare different labels.

        :param pred_methods: The prediction method that should be used.
        :type pred_methods: list of str
        :param what: Required, which data is plotted. Options are 'all', 'test', 'train'.
        :type what: str
        :param plt_labels: Adds subplot with labels.
        :type plt_labels: bool
        :param figsize: Sets figure size of plot.
        :type figsize: tuple
        :param perplexity: Optional, default 30. The perplexity parameter for TSNE.
        :type perplexity: int
        :param as_cols: Optional, default False. If True subplots are arranged in columns.
        :type as_cols: bool
        :param rdseed: Optional, default None. Random seed for numpy random.
        :type rdseed: int
        :param dot_size: Optional, default 5. Size of the point in the scatter plot.
        :type dot_size: int
        :param verb: Optional, default False Additional output is printed.
        :type verb: bool
        """
        if type(rdseed) == int:
            np.random.seed(seed=rdseed)  # fixing random seed
        elif rdseed is not None:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Seed has to be of type int.")

        if type(dot_size) != int:
            dot_size = 5
            print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                  "Value of 'dot_size' has to be of type int. It is set to 5.")

        if type(pred_methods) is not list and type(pred_methods) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type list or string.")

        if type(pred_methods) is str:
            pred_methods = [pred_methods]

        for m in pred_methods:
            if m not in self.predictions.keys():
                raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                                 "Prediction method {} is not in the predictions dictionary.\n".format(m) +
                                 "Valid options are: {}".format(self.predictions.keys()))

        nrows = len(pred_methods)
        ncols = 1
        subtitles = [''] * nrows

        if plt_labels:
            subtitles = ['Labels'] + pred_methods
            nrows = nrows + 1  # take the true labels into account

        # switch rows and cols
        if as_cols:
            temp = nrows
            nrows = ncols
            ncols = temp

        if what not in ['all', 'test', 'train']:
            what = 'all'
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")

        # -------- MATPLOTLIB event handler --------

        def update_annot(ind):
            for i in range(nrows * ncols):
                pos = sc[i].get_offsets()[ind['ind'][0]]
                annot[i].xy = pos
                if ind['ind'].size == 1:
                    id = ind['ind'][0]

                    text = "{}, {}".format(self.files[self.get_file_nbrs(what)[id]].split('/')[-1].split('-')[0],
                                           self.get_event_nbrs(what)[id])
                    if plt_labels:
                        text = text + \
                               ", {}".format(
                                   self.labels[self.get_label_nbrs(what, verb=verb)[id]])
                else:
                    text = "{}".format(
                        " ".join(list(map(str, [self.get_event_nbrs(what)[id] for id in ind['ind']]))))
                annot[i].set_text(text)
                annot[i].get_bbox_patch().set_alpha(0.7)

        def hover(event):
            for i in range(nrows * ncols):
                vis = annot[i].get_visible()
                if event.inaxes == ax[i]:
                    # cont: bool, whether it contains something or not
                    cont, ind = sc[i].contains(event)
                    # print(ind)
                    if cont:
                        update_annot(ind)
                        annot[i].set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot[i].set_visible(False)
                            fig.canvas.draw_idle()

        def onclick(event):
            for i in range(nrows * ncols):
                if event.inaxes == ax[i]:
                    _, ind = sc[i].contains(event)
                    if ind['ind'].size > 1:
                        print('Select a single event.')
                    elif ind['ind'].size == 1:
                        id = ind['ind'][0]

                        text = "{}, {}".format(
                            self.files[self.get_file_nbrs(what, verb=verb)[
                                id]].split('/')[-1].split('-')[0],
                            self.get_event_nbrs(what, verb=verb)[id])
                        if plt_labels:
                            text = text + ", {} = {}".format(self.labels[self.get_label_nbrs(what, verb=verb)[id]],
                                                             self.get_label_nbrs(what, verb=verb)[id])

                        print("Plotting Event nbr. '{}' from file '{}'.".format(
                            self.get_event_nbrs(what, verb=verb)[id],
                            self.files[self.get_file_nbrs(what, verb=verb)[id]]))

                        self.plot_event(id, what, False, text, verb)

        def on_key(event):
            if event.key == 'm':
                for i in range(nrows * ncols):
                    if event.inaxes == ax[i]:
                        _, ind = sc[i].contains(event)
                        if ind['ind'].size > 1:
                            print('Select a single event.')
                        elif ind['ind'].size == 1:
                            id = ind['ind'][0]

                            text = "{}, {}".format(
                                self.files[self.get_file_nbrs(what, verb=verb)[
                                    id]].split('/')[-1].split('-')[0],
                                self.get_event_nbrs(what, verb=verb)[id])
                            if plt_labels:
                                text = text + ", {} = {}".format(
                                    self.labels[self.get_label_nbrs(
                                        what, verb=verb)[id]],
                                    self.get_label_nbrs(what, verb=verb)[id])

                            print("Plotting Event nbr. '{}' from file '{}'.".format(
                                self.get_event_nbrs(what, verb=verb)[id],
                                self.files[self.get_file_nbrs(what, verb=verb)[id]]))

                            self.plot_event(id, what, True, text, verb)

            elif event.key == 'p':
                for i in range(nrows * ncols):
                    if event.inaxes == ax[i]:
                        _, ind = sc[i].contains(event)
                        if ind['ind'].size > 1:
                            print('Select a single event.')
                        elif ind['ind'].size == 1:
                            id = ind['ind'][0]

        if self.save_as == False:
            print("-------------------------------------------------------------------------")
            print("Hovering over an event shows you the event number.")
            print("When clicking on a single event a window with its timeseries is opened.")
            print("Hovering over a a single event and pressing 'm' also opnes the timeseries")
            print("of this event and adds the calculated mainparameters to the plot.")
            print("-------------------------------------------------------------------------")

        # -------- PLOT --------
        # TSNE
        princcomp = TSNE(n_components=2, perplexity=perplexity).fit_transform(
            self.get_features(what, verb=verb))

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()

        if type(figsize) is not tuple:
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(
                # nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
                nrows=nrows, ncols=ncols, sharex=True, sharey=True)

        annot = [None] * nrows * ncols
        sc = [None] * nrows * ncols

        if nrows * ncols == 1:
            ax = [ax]

        start_i = 0
        if plt_labels:
            start_i = 1
            ax[0].set_title(subtitles[0])
            _, _, leg = self.convert_to_labels_colors(self.get_label_nbrs(what, verb=verb), return_legend=True,
                                                      verb=verb)

            if self.save_as != False:
                sc[0] = ax[0].scatter(princcomp[:, 0], princcomp[:, 1],
                                      c=self.get_labels_in_color(what=what, verb=verb), s=dot_size, alpha=0.7)
            else:
                sc[0] = ax[0].scatter(princcomp[:, 0], princcomp[:, 1],
                                      c=self.get_labels_in_color(what=what, verb=verb), s=dot_size, alpha=0.7)
            pop = [None] * leg.shape[1]
            for i in range(leg.shape[1]):
                # pop[i] = mpl.patches.Patch(color=leg[2,i], label="{} ({})".format(leg[1,i], leg[0,i]))
                pop[i] = mpl.patches.Patch(
                    color=leg[2, i], label="{} ({})".format(leg[0, i], leg[1, i]))
            ax[0].legend(handles=pop, framealpha=0.3)

        for i in range(start_i, nrows * ncols):
            ax[i].set_title(subtitles[i])

            if self.save_as != False:
                sc[i] = ax[i].scatter(princcomp[:, 0],
                                      princcomp[:, 1],
                                      c=self.get_pred_in_color(pred_methods[i - start_i],
                                                               what=what, verb=verb),
                                      s=dot_size,
                                      alpha=0.7)
            else:
                sc[i] = ax[i].scatter(princcomp[:, 0],
                                      princcomp[:, 1],
                                      c=self.get_pred_in_color(pred_methods[i - start_i],
                                                               what=what, verb=verb),
                                      s=dot_size,
                                      alpha=0.7)

        for i in range(nrows * ncols):
            annot[i] = ax[i].annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
            annot[i].set_visible(False)
            ax[i].tick_params(left=False,
                              bottom=False,
                              labelleft=False,
                              labelbottom=False)

        fig.tight_layout()
        plt.tight_layout()
        if self.save_as != False:
            if plt_labels:
                plt.gcf().subplots_adjust(top=0.95, right=0.5)
                ax[0].legend(handles=pop, framealpha=0.3,
                             loc='center left', bbox_to_anchor=(1.0, 0.5))

            # set_mpl_backend_fontsize(10)
            if pred_methods == [] and plt_labels:
                # plt.savefig(
                #     '{}tsne-{}.pgf'.format(self.save_plot_dir, 'labels'))
                plt.savefig(
                    '{}tsne-{}.{}'.format(self.save_plot_dir, 'labels', self.save_as))
            else:
                # plt.savefig(
                #     '{}tsne-{}.pgf'.format(self.save_plot_dir, '_'.join(pred_methods)))
                plt.savefig(
                    '{}tsne-{}.{}'.format(self.save_plot_dir, '_'.join(pred_methods), self.save_as))
        else:
            fig.canvas.mpl_connect('key_press_event', on_key)
            fig.canvas.mpl_connect("motion_notify_event", hover)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        plt.close()

    def plt_pred_with_pca(self, pred_methods, xy_comp=(1, 2), what='all', plt_labels=True,
                          figsize=None, as_cols=False, rdseed=None, dot_size=5,
                          verb=False):
        """
        Plots data with PCE when given a one or a list of predictions method
        to compare different labels.

        :param pred_methods: The prediction method that should be used.
        :type pred_methods: list of str
        :param xy_comp: Optional, default (1,2) Select with pc's are used for x and y axis.
        :type xy_comp: tuple
        :param what: Required, which data is plotted. Options are 'all', 'test', 'train'.
        :type what: str
        :param plt_labels: Adds subplot with labels.
        :type plt_labels: bool
        :param figsize: Sets figure size of plot.
        :type figsize: tuple
        :param as_cols: Optional, default False. If True subplots are arranged in columns.
        :type as_cols: bool
        :param rdseed: Optional, default None. Random seed for numpy random.
        :type rdseed: int
        :param dot_size: Optional, default 5. Size of the point in the scatter plot.
        :type dot_size: int
        :param verb: Optional, default False Additional output is printed.
        :type verb: bool
        """
        if type(rdseed) == int:
            np.random.seed(seed=rdseed)  # fixing random seed
        elif rdseed is not None:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Seed has to be of type int.")

        if type(xy_comp) != tuple and \
                len(xy_comp) != 2 and \
                type(xy_comp[0]) != int and \
                type(xy_comp[1]) != int and \
                xy_comp[0] > 0 and \
                xy_comp[1] > 0:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "The parameter xy_comp has to be of shape '(<int>,<int>)'" +
                             "with the integer being > 0.")

        if type(dot_size) != int:
            dot_size = 5
            print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                  "Value of 'dot_size' has to be of type int. It is set to 5.")

        if type(pred_methods) is not list and type(pred_methods) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type list or string.")

        if type(pred_methods) is str:
            pred_methods = [pred_methods]

        for m in pred_methods:
            if m not in self.predictions.keys():
                raise ValueError(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                                 "Prediction method {} is not in the predictions dictionary.\n".format(m) +
                                 "Valid options are: {}".format(self.predictions.keys()))

        nrows = len(pred_methods)
        ncols = 1
        subtitles = [''] * nrows

        if plt_labels:
            subtitles = ['Labels'] + pred_methods
            nrows = nrows + 1  # take the true labels into account

        # switch rows and cols
        if as_cols:
            temp = nrows
            nrows = ncols
            ncols = temp

        if what not in ['all', 'test', 'train']:
            what = 'all'
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")

        # -------- MATPLOTLIB event handler --------

        def update_annot(ind):
            for i in range(nrows * ncols):
                pos = sc[i].get_offsets()[ind['ind'][0]]
                annot[i].xy = pos
                if ind['ind'].size == 1:
                    id = ind['ind'][0]

                    text = "{}, {}".format(self.files[self.get_file_nbrs(what)[id]].split('/')[-1].split('-')[0],
                                           self.get_event_nbrs(what)[id])
                    if plt_labels:
                        text = text + \
                               ", {}".format(
                                   self.labels[self.get_label_nbrs(what, verb=verb)[id]])
                else:
                    text = "{}".format(
                        " ".join(list(map(str, [self.get_event_nbrs(what)[id] for id in ind['ind']]))))
                annot[i].set_text(text)
                annot[i].get_bbox_patch().set_alpha(0.7)

        def hover(event):
            for i in range(nrows * ncols):
                vis = annot[i].get_visible()
                if event.inaxes == ax[i]:
                    # cont: bool, whether it contains something or not
                    cont, ind = sc[i].contains(event)
                    # print(ind)
                    if cont:
                        update_annot(ind)
                        annot[i].set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot[i].set_visible(False)
                            fig.canvas.draw_idle()

        def onclick(event):
            for i in range(nrows * ncols):
                if event.inaxes == ax[i]:
                    _, ind = sc[i].contains(event)
                    if ind['ind'].size > 1:
                        print('Select a single event.')
                    elif ind['ind'].size == 1:
                        id = ind['ind'][0]

                        text = "{}, {}".format(
                            self.files[self.get_file_nbrs(what, verb=verb)[
                                id]].split('/')[-1].split('-')[0],
                            self.get_event_nbrs(what, verb=verb)[id])
                        if plt_labels:
                            text = text + ", {} = {}".format(self.labels[self.get_label_nbrs(what, verb=verb)[id]],
                                                             self.get_label_nbrs(what, verb=verb)[id])

                        print("Plotting Event nbr. '{}' from file '{}'.".format(
                            self.get_event_nbrs(what, verb=verb)[id],
                            self.files[self.get_file_nbrs(what, verb=verb)[id]]))

                        self.plot_event(id, what, False, text, verb)

        def on_key(event):
            if event.key == 'm':
                for i in range(nrows * ncols):
                    if event.inaxes == ax[i]:
                        _, ind = sc[i].contains(event)
                        if ind['ind'].size > 1:
                            print('Select a single event.')
                        elif ind['ind'].size == 1:
                            id = ind['ind'][0]

                            text = "{}, {}".format(
                                self.files[self.get_file_nbrs(what, verb=verb)[
                                    id]].split('/')[-1].split('-')[0],
                                self.get_event_nbrs(what, verb=verb)[id])
                            if plt_labels:
                                text = text + ", {} = {}".format(
                                    self.labels[self.get_label_nbrs(
                                        what, verb=verb)[id]],
                                    self.get_label_nbrs(what, verb=verb)[id])
                            print("Plotting Event nbr. '{}' from file '{}'.".format(
                                self.get_event_nbrs(what, verb=verb)[id],
                                self.files[self.get_file_nbrs(what, verb=verb)[id]]))

                            self.plot_event(id, what, True, text, verb)

            elif event.key == 'p':
                for i in range(nrows * ncols):
                    if event.inaxes == ax[i]:
                        _, ind = sc[i].contains(event)
                        if ind['ind'].size > 1:
                            print('Select a single event.')
                        elif ind['ind'].size == 1:
                            id = ind['ind'][0]

        if self.save_as == False:
            print("-------------------------------------------------------------------------")
            print('Hovering over an event shows you the event number.')
            print('When clicking on a single event a window with its timeseries is opened.')
            print("Hovering over a a single event and pressing 'm' also opnes the timeseries")
            print('of this event and adds the calculated mainparameters to the plot.')
            print('-------------------------------------------------------------------------')

        # -------- PLOT --------
        # PCA
        pca = PCA(n_components=np.max(xy_comp) + 1)
        princcomp = pca.fit_transform(self.get_features(what, verb=verb))
        princcomp = princcomp[:, (xy_comp[0], xy_comp[1])]
        print('Explained Variance: ', pca.explained_variance_ratio_)
        print('Singular Values: ', pca.singular_values_)

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()

        if type(figsize) is not tuple:
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)

        annot = [None] * nrows * ncols
        sc = [None] * nrows * ncols

        if nrows * ncols == 1:
            ax = [ax]

        start_i = 0
        if plt_labels:
            start_i = 1
            ax[0].set_title(subtitles[0])
            _, _, leg = self.convert_to_labels_colors(self.get_label_nbrs(what, verb=verb), return_legend=True,
                                                      verb=verb)

            if self.save_as != False:
                sc[0] = ax[0].scatter(princcomp[:, 0], princcomp[:, 1],
                                      c=self.get_labels_in_color(what=what, verb=verb), s=dot_size, alpha=0.7)
            else:
                sc[0] = ax[0].scatter(princcomp[:, 0], princcomp[:, 1],
                                      c=self.get_labels_in_color(what=what, verb=verb), s=dot_size, alpha=0.7)
            pop = [None] * leg.shape[1]
            for i in range(leg.shape[1]):
                # pop[i] = mpl.patches.Patch(color=leg[2,i], label="{} ({})".format(leg[1,i], leg[0,i]))
                pop[i] = mpl.patches.Patch(
                    color=leg[2, i], label="{} ({})".format(leg[0, i], leg[1, i]))
            ax[0].legend(handles=pop, framealpha=0.3)

        for i in range(start_i, nrows * ncols):
            ax[i].set_title(subtitles[i])

            if self.save_as != False:
                sc[i] = ax[i].scatter(princcomp[:, 0],
                                      princcomp[:, 1],
                                      c=self.get_pred_in_color(pred_methods[i - start_i],
                                                               what=what, verb=verb),
                                      s=dot_size,
                                      alpha=0.7)
            else:
                sc[i] = ax[i].scatter(princcomp[:, 0],
                                      princcomp[:, 1],
                                      c=self.get_pred_in_color(pred_methods[i - start_i],
                                                               what=what, verb=verb),
                                      s=dot_size,
                                      alpha=0.7)

        for i in range(nrows * ncols):
            annot[i] = ax[i].annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
            annot[i].set_visible(False)

        fig.tight_layout()
        plt.tight_layout()
        if self.save_as != False:
            if plt_labels:
                plt.gcf().subplots_adjust(top=0.95, right=0.5)
                ax[0].legend(handles=pop, framealpha=0.3,
                             loc='center left', bbox_to_anchor=(1.0, 0.5))

            set_mpl_backend_fontsize(10)
            if pred_methods == [] and plt_labels:
                # plt.savefig(
                #     '{}tsne-{}.pgf'.format(self.save_plot_dir, 'labels'))
                plt.savefig(
                    '{}pca-{}.{}'.format(self.save_plot_dir, 'labels', self.save_as))
            else:
                # plt.savefig(
                #     '{}tsne-{}.pgf'.format(self.save_plot_dir, '_'.join(pred_methods)))
                plt.savefig(
                    '{}pca-{}.{}'.format(self.save_plot_dir, '_'.join(pred_methods), self.save_as))
        else:
            fig.canvas.mpl_connect('key_press_event', on_key)
            fig.canvas.mpl_connect("motion_notify_event", hover)
            fig.canvas.mpl_connect('button_press_event', onclick)

            plt.show()
        plt.close()

    def pulse_height_histogram(self,
                               ncols=2,
                               extend_plot=False,
                               figsize=None,
                               bins='auto',
                               verb=False):
        """
        Plots a histogram for all labels of the pulse hights in different
        subplots.

        :param ncols: Optional, default 2. Number of plots side by side.
        :type ncols: int
        :param extend_plot: Optional, default False. Sets the x axis of all histograms to the same limits.
        :type extend_plot: bool
        :param figsize: Optional, default None. Changes the overall figure size.
        :type figsize: tuple
        :param bins: Optional, default auto. Bins for the histograms.
        :type bins: int
        :param verb: Optional, default False. Ouputs additional information.
        :type verb: bool
        """
        max_height = self.get_mainpar(
            verb=verb)[:, self.mainpar_labels['pulse_height']]

        max_max_height = np.max(max_height)
        min_max_height = np.min(max_height)

        unique_label_nbrs = np.unique(self.get_label_nbrs(verb=verb))
        max_height_per_label = dict(
            [(l, max_height[self.get_label_nbrs(verb=verb) == l]) for l in unique_label_nbrs])

        nrows = math.ceil(len(unique_label_nbrs) / ncols)

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        for i, l in enumerate(unique_label_nbrs):
            j = i % ncols
            k = int(i / ncols)
            # print("i ({}); j ({}); k ({})".format(i,j,k))
            if k * ncols + j > i:
                break
            ax[k][j].set_title(self.labels[l])
            ax[k][j].hist(max_height_per_label[l], bins=bins)

        fig.tight_layout()
        plt.tight_layout()
        if extend_plot:
            for i, l in enumerate(unique_label_nbrs):
                j = i % ncols
                k = int(i / ncols)
                # print("i ({}); j ({}); k ({})".format(i,j,k))
                if k * ncols + j > i:
                    break
                ax[k][j].set_xlim(min_max_height, max_max_height)

        if self.save_as != False:
            # plt.savefig('{}labels_dist.pgf'.format(self.save_plot_dir))
            plt.savefig('{}labels_dist.{}'.format(
                self.save_plot_dir, self.save_as))
        else:
            plt.show()
        plt.close()

    def events_saturated_histogram(self, figsize=None, bins='auto', verb=False, ylog=False):
        """
        Plots a histogram for all event pulses and strongly saturated event pulses
        in a single plot.

        :param figsize: Optional, default None. Changes the overall figure size.
        :type figsize: tuple
        :param bins: Optional, default auto. Bins for the histograms.
        :type bins: int
        :param ylog: Optional, default False. If True the y axis is in log scale.
        :type ylog: bool
        """

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()
            mpl.use('pdf')

        max_height = self.get_mainpar(
            verb=verb)[:, self.mainpar_labels['pulse_height']]

        unique_label_nbrs = np.unique(self.get_label_nbrs(verb=verb))
        max_height_per_label = dict(
            [(l, max_height[self.get_label_nbrs(verb=verb) == l]) for l in unique_label_nbrs])

        events_nbr = list(self.labels.keys())[list(
            self.labels.values()).index('Event_Pulse')]
        saturated_nbr = list(self.labels.keys())[list(
            self.labels.values()).index('Strongly_Saturated_Event_Pulse')]

        if events_nbr and saturated_nbr in max_height_per_label:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

            ax.hist([max_height_per_label[events_nbr], max_height_per_label[saturated_nbr]],
                    label=['{}'.format(self.labels[events_nbr]), '{}'.format(
                        self.labels[saturated_nbr])],
                    bins=bins,
                    histtype='barstacked')

            ax.set_xlabel('pulse height (V)')
            ax.set_ylabel('number of events')

            plt.legend()
            fig.tight_layout()
            plt.tight_layout()
            if ylog:
                ax.set_yscale('log')
            if self.save_as != False:
                if ylog:
                    plt.savefig(
                        '{}evt_sat_hist-ylog.{}'.format(self.save_plot_dir, self.save_as))
                else:
                    plt.savefig('{}evt_sat_hist.{}'.format(
                        self.save_plot_dir, self.save_as))
            else:
                plt.show()
            plt.close()
        else:
            raise KeyError(
                'No Labels for Event Pulses or Strongly Saturated Event Pulses')

    def correctly_labeled_per_v(self, pred_method, what='all', bin_size=4, ncols=2, figsize=None, extend_plot=False,
                                verb=False):
        """
        Plots the number of correctly predicted labels over volts (pulse height)
        for every label.

        :param pred_method: Required. Name of the predictions method.
        :type pred_method: str
        :param what: Optional, default all. Test or train data or all.
        :type what: str
        :param bin_size: Optional, default 4. Bin size for calculating the average.
        :type bin_size: int
        :param ncols: Optional, default 2. Number of plots side by side.
        :type ncols: int
        :param figsize: Optional, default None. Size of the figure for matplotlib.
        :type figsize: tuple
        :param extend_plot: Optional, default False. If True x limits is set to the same for all subplots.
        :type extend_plot: bool
        :param verb: Optional, default False. If True additional information is printed on the console.
        :type verb: bool
        """

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()

        if type(pred_method) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type string.")
            return None

        if pred_method not in self.predictions.keys():
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Prediction method {} is not in the predictions dictionary.\n".format(pred_method) +
                             "Valid options are: {}".format(self.predictions.keys()))

        if not self.get_pred_true_labels(pred_method):
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "Only prediction methods with the same labeling as the" +
                      "correct labels are supported by this function.")
            return

        max_height = self.get_mainpar(what, verb=verb)[
                     :, self.mainpar_labels['pulse_height']]

        max_max_height = np.max(max_height)
        min_max_height = np.min(max_height)

        unique_label_nbrs, unique_label_counts = np.unique(
            self.get_label_nbrs(what, verb=verb), return_counts=True)
        data_dict_sorted = dict(
            [(l, ([None] * c, [None] * c, [None] * c)) for l, c in zip(unique_label_nbrs, unique_label_counts)])

        pos_counter = dict([(l, 0) for l in unique_label_nbrs])

        for h, tl, pl in sorted(
                zip(max_height, self.get_label_nbrs(what, verb=verb), self.get_pred(pred_method, what, verb=verb))):
            data_dict_sorted[tl][0][pos_counter[tl]] = h
            data_dict_sorted[tl][1][pos_counter[tl]] = pl
            pos_counter[tl] += 1

        for l in unique_label_nbrs:
            pl = np.array(data_dict_sorted[l][1])
            pl[pl != l] = 0
            pl[pl == l] = 1
            # pl = pl/pl.shape[0]
            # pl = np.cumsum(pl).tolist()
            for i, c in enumerate(pl):
                data_dict_sorted[l][2][i] = c

        bin_boundries = {}
        for l in unique_label_nbrs:
            bins = math.ceil(len(data_dict_sorted[l][2]) / bin_size)
            bin_boundries[l] = ([None] * bins, [None] * bins)
            for i in range(bins):
                upper = (i + 1) * bin_size if (i + 1) * bin_size < len(data_dict_sorted[l][2]) else len(
                    data_dict_sorted[l][2])
                lower = i * bin_size
                bin_boundries[l][0][i] = np.mean(
                    data_dict_sorted[l][0][lower:upper])
                bin_boundries[l][1][i] = np.mean(
                    data_dict_sorted[l][2][lower:upper])

        nrows = math.ceil(len(unique_label_nbrs) / ncols)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        for i, l in enumerate(unique_label_nbrs):
            j = i % ncols
            k = int(i / ncols)
            if k * ncols + j > i:
                break
            ax[k][j].set_title(self.labels[l])
            # ax[k][j].hist(data_dict_sorted[l][0],
            #               bins=bin_boundries[l][0])
            if extend_plot:
                ax[k][j].plot([min_max_height] + bin_boundries[l][0] + [max_max_height],
                              [0] + bin_boundries[l][1] + [bin_boundries[l][1][-1]])
            else:
                ax[k][j].plot(bin_boundries[l][0],
                              bin_boundries[l][1])
            # ax[k][j].set_xlable('pulse height (V)')
            # ax[k][j].set_ylable('accuray')

        fig.tight_layout()
        # plt.xlable('pulse height (V)')
        # plt.ylable('accuray')
        # plt.show()
        plt.tight_layout()
        if self.save_as != False:
            plt.savefig('{}correctly_labeled_per_v.{}'.format(
                self.save_plot_dir, self.save_as))
        else:
            plt.show()
        plt.close()

    def correctly_labeled_events_per_pulse_height(self, pred_method, what='all',
                                                  bin_size=4, ncols=2, extend_plot=False,
                                                  figsize=None, verb=False):
        """
        Plots the number of correctly predicted labels over volts (pulse height)
        for events.

        :param pred_method: Required. Name of the predictions method.
        :type pred_method: str
        :param what: Optional, default all. Test or train data or all.
        :type what: str
        :param bin_size: Optional, default 4. Bin size for calculating the average.
        :type bin_size: int
        :param ncols: Optional, default 2. Number of plots side by side.
        :type ncols:  int
        :param extend_plot: Optional, default False. If True x limits is set to the same for all subplots.
        :type extend_plot: bool
        :param figsize: Optional, default None. Changes the overall figure size.
        :type figsize: tuple
        :param verb: Optional, default False. If True additional information is printed on the console.
        :type verb: bool
        """

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()

        if type(pred_method) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type string.")
            return None

        if pred_method not in self.predictions.keys():
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Prediction method {} is not in the predictions dictionary.\n".format(pred_method) +
                             "Valid options are: {}".format(self.predictions.keys()))

        if not self.get_pred_true_labels(pred_method):
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "Only prediction methods with the same labeling as the" +
                      "correct labels are supported by this function.")
            return

        max_height = self.get_mainpar(what, verb=verb)[
                     :, self.mainpar_labels['pulse_height']]

        unique_label_nbrs, unique_label_counts = np.unique(
            self.get_label_nbrs(what, verb=verb), return_counts=True)
        data_dict_sorted = dict(
            [(l, ([None] * c, [None] * c, [None] * c)) for l, c in zip(unique_label_nbrs, unique_label_counts)])

        pos_counter = dict([(l, 0) for l in unique_label_nbrs])

        for h, tl, pl in sorted(
                zip(max_height, self.get_label_nbrs(what, verb=verb), self.get_pred(pred_method, what, verb=verb))):
            data_dict_sorted[tl][0][pos_counter[tl]] = h
            data_dict_sorted[tl][1][pos_counter[tl]] = pl
            pos_counter[tl] += 1

        for l in unique_label_nbrs:
            pl = np.array(data_dict_sorted[l][1])
            pl[pl != l] = 0
            pl[pl == l] = 1
            # pl = pl/pl.shape[0]
            # pl = np.cumsum(pl).tolist()
            for i, c in enumerate(pl):
                data_dict_sorted[l][2][i] = c

        bin_boundries = {}
        for l in unique_label_nbrs:
            bins = math.ceil(len(data_dict_sorted[l][2]) / bin_size)
            bin_boundries[l] = ([None] * bins, [None] * bins)
            for i in range(bins):
                upper = (i + 1) * bin_size if (i + 1) * bin_size < len(data_dict_sorted[l][2]) else len(
                    data_dict_sorted[l][2])
                lower = i * bin_size
                bin_boundries[l][0][i] = np.mean(
                    data_dict_sorted[l][0][lower:upper])
                bin_boundries[l][1][i] = np.mean(
                    data_dict_sorted[l][2][lower:upper])

        l = 1  # events
        # nrows = math.ceil(len(unique_label_nbrs)/ncols)
        if figsize is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_title(self.labels[l])
        # ax[k][j].hist(data_dict_sorted[l][0],
        #               bins=bin_boundries[l][0])
        ax.plot(bin_boundries[l][0],
                bin_boundries[l][1])

        ax.set_xlabel('pulse height (V)')
        ax.set_ylabel('accuray')
        ax.set_xlim(0, 0.12)
        fig.tight_layout()
        plt.tight_layout()
        # plt.gcf().subplots_adjust(bottom=-0.1)
        if self.save_as != False:
            # plt.savefig(
            #     '{}correctly_labeled_events-{}.pgf'.format(self.save_plot_dir, pred_method))
            plt.savefig(
                '{}correctly_labeled_events-{}.{}'.format(self.save_plot_dir, pred_method, self.save_as))
        else:
            plt.show()
        plt.close()

    def confusion_matrix_pred(self, pred_method, what='all', rotation_xticklabels=0,
                              force_xlabelnbr=False, figsize=None, fig_title=False, verb=False):
        """
        Plots a confusion matrix to better visualize which labels are better
        predicted by a certain prediction method.
        In the (i,j) position the number of i labels which are predicted a j
        are written.
        When clicking on a matrix element the event number and from which file
        is printed out in the console

        :param pred_method: Required. Name of the predictions method.
        :type pred_method:  str
        :param what: Optional, default all. Test or train data or all.
        :type what: str
        :param rotation_xticklabels: Optional, default 0. Lets you rotate the x tick labels.
        :type rotation_xticklabels: int
        :param force_xlabelnbr: Optional, default False. Uses the number instead of the labels for better readability.
        :type force_xlabelnbr: bool
        :param figsize: Optional, default None. Changes the overall figure size.
        :type figsize: tuple
        :param verb: Optional, default False. If True additional information is printed on the console.
        :type verb: bool
        """

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()

        if type(pred_method) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type string.")
            return None

        if type(fig_title) != bool:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "The parameter 'fig_title' must be True or False.\n")

        if pred_method not in self.predictions.keys():
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Prediction method {} is not in the predictions dictionary.\n".format(pred_method) +
                             "Valid options are: {}".format(self.predictions.keys()))

        if self.get_pred_true_labels(pred_method):
            ylabels_order = [self.labels[l] + " ({})".format(int(l))
                             for l in np.unique(np.hstack(
                    [np.unique(self.get_pred(pred_method, what, verb=verb)),
                     np.unique(self.get_label_nbrs(what, verb=verb))])).tolist()]
        else:
            ylabels_order = [self.labels[l] + " ({})".format(int(l))
                             for l in np.unique(self.get_label_nbrs(what, verb=verb))]

        if self.get_pred_true_labels(pred_method):
            xlabels_order = ylabels_order
        else:
            xlabels_order = np.unique(self.get_pred(
                pred_method, what, verb=verb)).tolist()

        diff = len(ylabels_order) - len(xlabels_order)
        for i in range(abs(diff)):
            if diff > 0:  # len(ylabels_order)>len(xlabels_order)
                xlabels_order.append(chr(ord('a') + i))
            elif diff < 0:  # len(ylabels_order)<len(xlabels_order)
                ylabels_order.append(chr(ord('a') + i))

        pred_labels = self.get_pred(pred_method, what, verb=verb)
        if not self.get_pred_true_labels(pred_method):
            pred_labels_dict = dict(
                [(pl, tl) for tl, pl in zip(np.unique(self.get_label_nbrs(what, verb=verb)), xlabels_order)])
            pred_labels = [pred_labels_dict[l] for l in pred_labels]

        if self.get_pred_true_labels(pred_method) and force_xlabelnbr:
            xlabels_order = [l.split(')')[0] for l in [l.split(
                '(')[-1] for l in xlabels_order]]

        if type(xlabels_order) is np.ndarray:
            xlabels_order = xlabels_order.tolist()
        if type(ylabels_order) is np.ndarray:
            ylabels_order = ylabels_order.tolist()

        if self.get_pred_true_labels(pred_method):
            true_label_nbrs = self.get_label_nbrs(what, verb=verb)
        else:
            cm_conv_labels = dict([(n, i) for i, n in enumerate(
                np.unique(self.get_label_nbrs(what, verb=verb)))])
            true_label_nbrs = [cm_conv_labels[l]
                               for l in self.get_label_nbrs(what, verb=verb)]

        cm = confusion_matrix(true_label_nbrs,
                              self.get_pred(pred_method, what, verb=verb))
        if verb:
            print(cm)

        def onclick(event):
            if event.inaxes == ax:
                # cont, ind = cax.contains(event)

                x = event.xdata
                y = event.ydata
                i = int(y + 0.5)
                j = int(x + 0.5)

                selected_label_nbr = np.unique(
                    self.get_label_nbrs(verb=verb))[i]
                if self.get_pred_true_labels(pred_method):
                    selected_pred_nbr = np.unique(
                        self.get_label_nbrs(verb=verb))[j]
                else:
                    selected_pred_nbr = np.unique(
                        self.get_pred(pred_method, verb=verb))[j]

                selection = np.logical_and(self.get_label_nbrs(what, verb) == selected_label_nbr,
                                           self.get_pred(pred_method, what, verb) == selected_pred_nbr)

                results = [
                    self.get_file_nbrs(what, verb)[selection],
                    self.get_event_nbrs(what, verb)[selection],
                    self.get_label_nbrs(what, verb)[selection],
                    self.get_pred(pred_method, what, verb)[selection]
                ]

                print("---------- Events labeled as {} ({}) but predicted as {} ----------".format(
                    self.labels[int(selected_label_nbr)], int(selected_label_nbr), int(selected_pred_nbr)))
                for fn, en, ln, pn in zip(*results):
                    print("'{}' \t Event_nbr:{:>4} \t Label:{:>3}, \t Prediction:{:>3}".format(self.files[fn], en, ln,
                                                                                               pn))
                print("--------------------")
                print()

        if type(figsize) is not tuple:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)

        cax = ax.matshow(cm, cmap='plasma', alpha=0.7)
        fig.colorbar(cax)

        if ylabels_order != []:
            ax.set_yticks(np.arange(len(ylabels_order)))
            ax.set_yticklabels(ylabels_order)

        if xlabels_order != []:
            ax.set_xticks(np.arange(len(xlabels_order)))
            ax.set_xticklabels(xlabels_order,
                               rotation=rotation_xticklabels)

        plt.gcf().subplots_adjust(left=0.5)

        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:d}'.format(z), ha='center', va='center')

        ax.yaxis.set_label_position('right')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Labels')
        if (fig_title):
            plt.title('{} - Confusion Matrix'.format(pred_method))
        fig.tight_layout()
        plt.tight_layout()
        if self.save_as == False:
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        else:
            # plt.savefig(
            #     '{}confMat-{}.pgf'.format(self.save_plot_dir, pred_method))
            plt.savefig(
                '{}confMat-{}.{}'.format(self.save_plot_dir, pred_method, self.save_as))

        plt.close()

    def plot_labels_distribution(self, figsize=None):
        """
        Uses a bar graph to visualize how often a label occures in the
        dataset

        :param figsize: Optional, default None. Changes the overall figure size.
        :type figsize: tuple
        """

        plt.close()
        if figsize is None:
            use_cait_style(fontsize=14, autolayout=False, dpi=None)
        else:
            use_cait_style(x_size=figsize[0], y_size=figsize[1], fontsize=14,
                           autolayout=False, dpi=None)

        if self.save_as == 'pgf':
            set_mpl_backend_pgf()

        lnbr, lcnt = np.unique(self.label_nbrs, return_counts=True)
        lenum = np.arange(lnbr.shape[0])

        if type(figsize) is not tuple:
            _, ax = plt.subplots()
        else:
            _, ax = plt.subplots(figsize=figsize)

        bars = plt.bar(lenum, lcnt)
        for i, b in enumerate(bars):
            b.set_color(self.color_order[i])
        plt.xticks(lenum, lnbr)

        rects = ax.patches
        for rect, i in zip(rects, lcnt):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 15, i,
                    ha='center', va='bottom')
        _, top = plt.ylim()
        plt.ylim(0, 1.1 * top)

        ax.set_ylabel("number of events")
        ax.set_xlabel("labels")

        pop = [None] * len(lnbr)
        for i, j in enumerate(lnbr):
            pop[i] = mpl.patches.Patch(color=self.color_order[i],
                                       label="{} ({})".format(self.labels[j], j))
        ax.legend(handles=pop, framealpha=0.3,
                  loc='center left', bbox_to_anchor=(1.0, 0.5))

        plt.gcf().subplots_adjust(right=0.5)
        if self.save_as != False:
            plt.savefig('{}labels_dist.{}'.format(
                self.save_plot_dir, self.save_as))
        else:
            plt.show()
        plt.close()

    def plt_pred_with_tsne_plotly(self, pred_methods, what='all',
                                  perplexity=30, rdseed=None, verb=False, inline=True):
        """
        Plots data with TSNE when given a one or a list of predictions method
        to compare different labels.

        :param pred_methods: Required. Prediction method that should be used.
        :type pred_methods: list
        :param what: Required. Which data is plotted, 'all', 'test' or 'train'.
        :type what: str
        :param perplexity: Optional, default 30. Perplexity parameter for TSNE.
        :type perplexity: int
        :param rdseed: Optional, default None. Random seed for numpy random.
        :type rdseed: int
        :param verb: Optional, default False. Additional output is printed.
        :type verb: bool
        :param inline: Activates the inline mode of the Dash server, recommended for usage with Jupyter Notebooks
        :type inline: bool
        """

        if type(rdseed) == int:
            np.random.seed(seed=rdseed)  # fixing random seed
        elif rdseed is not None:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Seed has to be of type int.")

        if type(pred_methods) is not list and type(pred_methods) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type list or string.")

        if type(pred_methods) is str:
            pred_methods = [pred_methods]

        for m in pred_methods:
            if m not in self.predictions.keys():
                raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                                 "Prediction method {} is not in the predictions dictionary.\n".format(m) +
                                 "Valid options are: {}".format(self.predictions.keys()))

        if what not in ['all', 'test', 'train']:
            what = 'all'
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")

        # -------- PLOT --------
        # TSNE
        princcomp = TSNE(n_components=2, perplexity=perplexity).fit_transform(
            self.get_features(what, verb=verb))

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = Dash(__name__, external_stylesheets=external_stylesheets)

        styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }

        def scatter_plot(pred_meth):
            label_nbrs = self.get_label_nbrs(what, verb=verb)
            labels = [self.labels[i] for i in label_nbrs]
            # import ipdb; ipdb.set_trace()
            df = pd.DataFrame({
                "id": np.arange(self.get_events(what, verb).shape[0]),
                "file": [self.files[i] for i in self.get_file_nbrs(what, verb)],
                "x": princcomp[:, 0],
                "y": princcomp[:, 1],
                "label_nbrs": label_nbrs,
                "labels": labels,
                "color": label_nbrs.astype(str)
            })
            fig = None
            if pred_meth is None:
                fig = px.scatter(df,
                                 x="x",
                                 y="y",
                                 color="labels",
                                 hover_name="labels",
                                 hover_data=["id", "file"],
                                 )
                fig.update_yaxes(visible=False)
                fig.update_layout(height=600)
            else:
                fig = make_subplots(rows=2, cols=1,
                                    shared_xaxes=True,
                                    shared_yaxes=True,
                                    vertical_spacing=0.02)

                unique_label_nbrs = np.unique([label_nbrs, self.get_pred(pred_meth, what=what, verb=verb)])
                # import ipdb; ipdb.set_trace()
                dict_colors = dict([[j, self.color_order[i]] for i, j in enumerate(unique_label_nbrs)])

                group = "1"
                group_title = "Labels"

                for i in np.unique(self.get_label_nbrs(what, verb)):
                    sep = self.get_label_nbrs(what, verb) == i  # seperating label numbers
                    fig.add_trace(go.Scattergl(
                        x=princcomp[sep, 0],
                        y=princcomp[sep, 1],
                        name=self.labels[i],
                        # color=label_nbrs,
                        mode='markers',
                        # marker=dict(color=label_nbrs, coloraxis="coloraxis")
                        marker=dict(color=dict_colors[i]),  # coloraxis="coloraxis")
                        legendgroup=group,
                        legendgrouptitle_text=group_title,
                        # hover_name="labels",
                        # hover_data=["id", "file"],
                        # hovertemplate=
                        #    '<b>%{labels}</b><br><br>' +
                        #    'index: %{text}<br>' +
                        #    'file: %{file}<br>' +
                        #    'x: %{x}<br>' +
                        #    'y: %{y}',
                        customdata=np.array([df['id'][sep]]).T,
                    ),
                        row=1,
                        col=1,
                    )

                # if not self.get_pred_true_labels(pred_meth):
                group = "2"
                group_title = "Prediction"

                for i in np.unique(self.get_pred(pred_meth, what=what, verb=verb)):
                    sep = self.get_label_nbrs(what, verb) == i  # seperating label numbers
                    name = self.labels[i] if self.get_pred_true_labels(pred_meth) else "{}".format(i)

                    fig.add_trace(go.Scattergl(
                        x=princcomp[sep, 0],
                        y=princcomp[sep, 1],
                        name=name,
                        # color=self.get_pred(pred_meth, what=what, verb=verb).astype(str),
                        mode='markers',
                        marker=dict(color=dict_colors[i]),  # coloraxis="coloraxis")
                        legendgroup=group,
                        legendgrouptitle_text=group_title,
                        # hover_name="labels",
                        # hover_data=["id", "file"],
                        # hovertemplate=
                        #    ["<b>{}</b><br>index: {}<br>file: %{file}<br>x: %{x}<br>y: %{y}".format(name, id) for id in df['id'][sep]],
                        customdata=np.array([df['id'][sep]]).T,
                    ),
                        row=2,
                        col=1,
                    )
                fig.update_layout(showlegend=True, height=800,
                                  hovermode='closest')  # coloraxis=dict(colorscale=dict_colors))
            return fig

        def click_event(clickData=None):
            if clickData is None:
                # clickData = {"points":[{"pointIndex":0}]}
                clickData = {"points": [{"customdata": [0]}]}
            # index = clickData['points'][0]['pointIndex']
            index = clickData['points'][0]['customdata'][0]
            event_nbr = self.get_event_nbrs(what, verb)[index]
            event = self.get_events(what)[index]
            fig_event = px.line(x=np.arange(event.shape[0]), y=event)
            fig_event.update_xaxes(title={'text': ''})
            fig_event.update_yaxes(title={'text': ''})
            return fig_event

        def click_data(clickData=None):
            if clickData is None:
                # clickData = {"points":[{"pointIndex":0}]}
                clickData = {"points": [{"customdata": [0]}]}
            # index = clickData['points'][0]['pointIndex']
            index = clickData['points'][0]['customdata'][0]
            event_nbr = self.get_event_nbrs(what, verb)[index]
            file = self.files[self.get_file_nbrs(what, verb)[index]]
            label_nbr = self.get_label_nbrs(what, verb)[index]
            label = self.labels[label_nbr]
            return """Index: {}
                      Event number: {}
                      Event type: {}, {}
                      File: {}
                   """.format(index, event_nbr, label, label_nbr, file)

        app.layout = html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='pred_meth-dropdown',
                        options=[{'label': i, 'value': i} for i in pred_methods],
                        value=None
                    ),
                ]),
                html.Div([
                    dcc.Graph(
                        id='scatter-graph',
                        figure=scatter_plot(None)
                    ),
                ]),
            ], style={'width': '60%', 'height': '70%', 'display': 'inline-block'}),

            html.Div(className='row', children=[
                html.Div([
                    # dcc.Markdown("""
                    #     **Click Data**
                    #
                    #     Click on points in the graph.
                    # """),
                    # html.Pre(id='click-data', style=styles['pre']),
                    html.Div(id='textarea-click-data',
                             children=[click_data()],
                             style={'whiteSpace': 'pre-line'}),
                ], className='three columns', style={'width': '100%'}),
                html.Div([
                    dcc.Graph(id='event-graph', figure=click_event()),
                ], className='three columns', style={'width': '100%'}),
            ], style={'width': '39%', 'float': 'right', 'display': 'inline-block'})
        ])

        # @app.callback(
        #     Output('click-data', 'children'),
        #     Input('scatter-graph', 'clickData'))
        # def display_click_data(clickData):
        #     return json.dumps(clickData, indent=2)

        @app.callback(
            Output('textarea-click-data', 'children'),
            Input('scatter-graph', 'clickData'))
        def display_click_data(clickData):
            return click_data(clickData)

        @app.callback(
            Output('event-graph', 'figure'),
            Input('scatter-graph', 'clickData'))
        def display_click_event(clickData):
            return click_event(clickData)

        @app.callback(
            Output('scatter-graph', 'figure'),
            Input('pred_meth-dropdown', 'value'))
        def display_scatter_plot(pred_meth):
            return scatter_plot(pred_meth)

        if inline:
            app.run_server(mode='inline')
        else:
            app.run_server(mode='external')

    def plt_pred_with_pca_plotly(self, pred_methods, xy_comp=(1, 2), what='all',
                                 rdseed=None, verb=False, inline=True):
        """
        Plots data with PCE when given a one or a list of predictions method
        to compare different labels.

        :param pred_methods: Required. Prediction method that should be used.
        :type pred_methods: list
        :param xy_comp: Optional, default (1,2). Select with pc's are used for x and y axis.
        :type xy_comp: tuple
        :param what: Optional, default 'all'. Which data is plotted.
        :type what: str
        :param rdseed: Optional, default None. Random seed for numpy random.
        :type rdseed: int
        :param verb: Optional, default False. Additional output is printed.
        :type verb: bool
        :param inline: Activates the inline mode of the Dash server, recommended for usage with Jupyter Notebooks
        :type inline: bool
        """

        if type(rdseed) == int:
            np.random.seed(seed=rdseed)  # fixing random seed
        elif rdseed is not None:
            raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                             "Seed has to be of type int.")

        if type(pred_methods) is not list and type(pred_methods) is not str:
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "'pred_methods' has to of type list or string.")

        if type(pred_methods) is str:
            pred_methods = [pred_methods]

        for m in pred_methods:
            if m not in self.predictions.keys():
                raise ValueError(console_colors.FAIL + "ERROR: " + console_colors.ENDC +
                                 "Prediction method {} is not in the predictions dictionary.\n".format(m) +
                                 "Valid options are: {}".format(self.predictions.keys()))

        if what not in ['all', 'test', 'train']:
            what = 'all'
            if verb:
                print(console_colors.OKBLUE + "NOTE: " + console_colors.ENDC +
                      "If the value of 'what' is not 'train' or 'test' then all are shown.")

        # -------- PLOT --------
        # PCA
        pca = PCA(n_components=np.max(xy_comp) + 1)
        princcomp = pca.fit_transform(self.get_features(what, verb=verb))
        princcomp = princcomp[:, (xy_comp[0], xy_comp[1])]

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = Dash(__name__, external_stylesheets=external_stylesheets)

        styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }

        def scatter_plot(pred_meth):
            label_nbrs = self.get_label_nbrs(what, verb=verb)
            labels = [self.labels[i] for i in label_nbrs]
            # import ipdb; ipdb.set_trace()
            df = pd.DataFrame({
                "id": np.arange(self.get_events(what, verb).shape[0]),
                "file": [self.files[i] for i in self.get_file_nbrs(what, verb)],
                "x": princcomp[:, 0],
                "y": princcomp[:, 1],
                "label_nbrs": label_nbrs,
                "labels": labels,
                "color": label_nbrs.astype(str)
            })
            fig = None
            if pred_meth is None:
                fig = px.scatter(df,
                                 x="x",
                                 y="y",
                                 color="labels",
                                 hover_name="labels",
                                 hover_data=["id", "file"],
                                 )
                fig.update_yaxes(visible=False)
                fig.update_layout(height=600)
            else:
                fig = make_subplots(rows=2, cols=1,
                                    shared_xaxes=True,
                                    shared_yaxes=True,
                                    vertical_spacing=0.02)

                unique_label_nbrs = np.unique([label_nbrs, self.get_pred(pred_meth, what=what, verb=verb)])
                # import ipdb; ipdb.set_trace()
                dict_colors = dict([[j, self.color_order[i]] for i, j in enumerate(unique_label_nbrs)])

                group = "1"
                group_title = "Labels"

                for i in np.unique(self.get_label_nbrs(what, verb)):
                    sep = self.get_label_nbrs(what, verb) == i  # seperating label numbers
                    fig.add_trace(go.Scattergl(
                        x=princcomp[sep, 0],
                        y=princcomp[sep, 1],
                        name=self.labels[i],
                        # color=label_nbrs,
                        mode='markers',
                        # marker=dict(color=label_nbrs, coloraxis="coloraxis")
                        marker=dict(color=dict_colors[i]),  # coloraxis="coloraxis")
                        legendgroup=group,
                        legendgrouptitle_text=group_title,
                        # hover_name="labels",
                        # hover_data=["id", "file"],
                        # hovertemplate=
                        #    '<b>%{labels}</b><br><br>' +
                        #    'index: %{text}<br>' +
                        #    'file: %{file}<br>' +
                        #    'x: %{x}<br>' +
                        #    'y: %{y}',
                        customdata=np.array([df['id'][sep]]).T,
                    ),
                        row=1,
                        col=1,
                    )

                # if not self.get_pred_true_labels(pred_meth):
                group = "2"
                group_title = "Prediction"

                for i in np.unique(self.get_pred(pred_meth, what=what, verb=verb)):
                    sep = self.get_label_nbrs(what, verb) == i  # seperating label numbers
                    name = self.labels[i] if self.get_pred_true_labels(pred_meth) else "{}".format(i)

                    fig.add_trace(go.Scattergl(
                        x=princcomp[sep, 0],
                        y=princcomp[sep, 1],
                        name=name,
                        # color=self.get_pred(pred_meth, what=what, verb=verb).astype(str),
                        mode='markers',
                        marker=dict(color=dict_colors[i]),  # coloraxis="coloraxis")
                        legendgroup=group,
                        legendgrouptitle_text=group_title,
                        # hover_name="labels",
                        # hover_data=["id", "file"],
                        # hovertemplate=
                        #    ["<b>{}</b><br>index: {}<br>file: %{file}<br>x: %{x}<br>y: %{y}".format(name, id) for id in df['id'][sep]],
                        customdata=np.array([df['id'][sep]]).T,
                    ),
                        row=2,
                        col=1,
                    )
                fig.update_layout(showlegend=True, height=800,
                                  hovermode='closest')  # coloraxis=dict(colorscale=dict_colors))
            return fig

        def click_event(clickData=None):
            if clickData is None:
                # clickData = {"points":[{"pointIndex":0}]}
                clickData = {"points": [{"customdata": [0]}]}
            # index = clickData['points'][0]['pointIndex']
            index = clickData['points'][0]['customdata'][0]
            event_nbr = self.get_event_nbrs(what, verb)[index]
            event = self.get_events(what)[index]
            fig_event = px.line(x=np.arange(event.shape[0]), y=event)
            fig_event.update_xaxes(title={'text': ''})
            fig_event.update_yaxes(title={'text': ''})
            return fig_event

        def click_data(clickData=None):
            if clickData is None:
                # clickData = {"points":[{"pointIndex":0}]}
                clickData = {"points": [{"customdata": [0]}]}
            # index = clickData['points'][0]['pointIndex']
            index = clickData['points'][0]['customdata'][0]
            event_nbr = self.get_event_nbrs(what, verb)[index]
            file = self.files[self.get_file_nbrs(what, verb)[index]]
            label_nbr = self.get_label_nbrs(what, verb)[index]
            label = self.labels[label_nbr]
            return """Index: {}
                      Event number: {}
                      Event type: {}, {}
                      File: {}
                   """.format(index, event_nbr, label, label_nbr, file)

        app.layout = html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='pred_meth-dropdown',
                        options=[{'label': i, 'value': i} for i in pred_methods],
                        value=None
                    ),
                ]),
                html.Div([
                    dcc.Graph(
                        id='scatter-graph',
                        figure=scatter_plot(None)
                    ),
                ]),
            ], style={'width': '60%', 'height': '70%', 'display': 'inline-block'}),

            html.Div(className='row', children=[
                html.Div([
                    # dcc.Markdown("""
                    #     **Click Data**
                    #
                    #     Click on points in the graph.
                    # """),
                    # html.Pre(id='click-data', style=styles['pre']),
                    html.Div(id='textarea-click-data',
                             children=[click_data()],
                             style={'whiteSpace': 'pre-line'}),
                ], className='three columns', style={'width': '100%'}),
                html.Div([
                    dcc.Graph(id='event-graph', figure=click_event()),
                ], className='three columns', style={'width': '100%'}),
            ], style={'width': '39%', 'float': 'right', 'display': 'inline-block'})
        ])

        # @app.callback(
        #     Output('click-data', 'children'),
        #     Input('scatter-graph', 'clickData'))
        # def display_click_data(clickData):
        #     return json.dumps(clickData, indent=2)

        @app.callback(
            Output('textarea-click-data', 'children'),
            Input('scatter-graph', 'clickData'))
        def display_click_data(clickData):
            return click_data(clickData)

        @app.callback(
            Output('event-graph', 'figure'),
            Input('scatter-graph', 'clickData'))
        def display_click_event(clickData):
            return click_event(clickData)

        @app.callback(
            Output('scatter-graph', 'figure'),
            Input('pred_meth-dropdown', 'value'))
        def display_scatter_plot(pred_meth):
            return scatter_plot(pred_meth)

        if inline:
            app.run_server(mode='inline')
        else:
            app.run_server(mode='external')
