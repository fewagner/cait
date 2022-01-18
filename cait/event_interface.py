# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
from .evaluation._color import console_colors
from .features._fem import get_elements, plot_S1
from .filter._of import filter_event
from .fit._templates import sev_fit_template
from .styles._plt_styles import use_cait_style, make_grid
from .fit._saturation import scaled_logistic_curve
from .fit._numerical_fit import template

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class EventInterface:
    """
    A class for the viewing and labeling of Events from HDF5 data sets.

    The Event interface is one of the core classes of the cait package. It is used to view raw data events and label
    them in order to train supervised machine learning models. The interface is with an interactive menu
    optimized for an efficient workflow.

    :param record_length: The number of samples in a record window.
    :type record_length: int
    :param sample_frequency: The record frequency of the measurement.
    :type sample_frequency: int
    :param nmbr_channels: The number of channels of the detector modules,
        typically 1, 2 (phonon/light) or 3 (including I-sticks or ring).
    :type nmbr_channels: int
    :param down: The downsample rate for viewing the events. This can later be adapted in the interactive menu.
    :type down: int
    :param dpi: Dots per inch for the plots.
    :type dpi: int
    :param run: The number of the measurement run. This is a optional argument, to identify a measurement with a
        given module uniquely. Providing this argument has no effect, but might be useful in case you start multiple
        DataHandlers at once, to stay organized.
    :type run: string or None
    :param module: The naming of the detector module. Optional argument, for unique identification of the physics data.
        Providing this argument has no effect, but might be useful in case you start multiple
        DataHandlers at once, to stay organized.
    :type module: string or None

    >>> ei = ai.EventInterface()
    Event Interface Instance created.
    >>> ei.load_h5(path='./', fname='test_001', channels=[0,1])
    Nmbr triggered events: 4
    Nmbr testpulses:  11
    Nmbr noise:  4
    HDF5 File loaded.
    >>> ei.create_labels_csv(path='./')
    """

    def __init__(self,
                 record_length: int = 16384,
                 sample_frequency: int = 25000,
                 nmbr_channels: int = 2,
                 down: int = 1,
                 dpi: int = None,
                 run: str = None,
                 module: str = None,
                 pre_trigger_region: float = 1 / 8,
                 ):
        self.nmbr_channels = nmbr_channels
        self.module = module
        self.run = run
        self.nmbr_channels = nmbr_channels
        self.record_length = record_length
        self.down = down
        self.sample_frequency = sample_frequency
        self.window_size = int(record_length / down)
        self.show_mp = False
        self.show_derivative = False
        self.show_triangulation = False
        self.std_thres = []
        for i in range(self.nmbr_channels):
            self.std_thres.append(0.001)
        self.only_wrong = False
        self.show_filtered = False
        self.sev = False
        self.arr = False
        self.saturation = False
        self.fit_models = None
        self.stdevents = None
        self.saturation_pars = None
        self.of = None
        self.subtract_offset = False
        self.labels = {}
        self.predictions = {}
        self.model_names = {}
        self.valid_types = ['events', 'testpulses', 'noise']
        if self.nmbr_channels == 2:
            self.channel_names = ['Phonon', 'Light']
        else:
            self.channel_names = [
                'Channel {}'.format(i) for i in range(nmbr_channels)
            ]
        self.xlim = None
        self.ylim = None
        self.dpi = dpi
        self.show_time = False
        self.window = False
        self.threshold = None
        self.show_threshold = False
        self.pre_trigger_region = pre_trigger_region

        print('Event Interface Instance created.')

    # ------------------------------------------------------------
    # INCLUDE THE DATA
    # ------------------------------------------------------------

    # Load in the hdf5 dataset
    def load_h5(self,
                path: str,
                fname: str,
                channels: list = None,
                appendix=True,
                which_to_label=['events']):
        """
        Load a hdf5 dataset to the event interface instance. This is typically done right after the declaration of a new instance.

        :param path: Path to the file folder. E.g. "data/" --> filepath "data/fname-[appendix].h5".
        :type path: string
        :param channels: The numbers of the channels that are included in the HDF5 file. The should be consistent with
            the appendix of the file name. If the file has no appendix, the numbering can be done arbitrarily, e.g. with
            [0, 1] for a two-channel detector module.
        :type channels: list of string
        :param fname: The file name without suffix, e.g. "test_001.h5" --> "test_001".
        :type fname: string
        :param appendix: If True the appendix generated from the gen_h5_from_rdt function is automatically
            appended to the fname string. Use this argument, if your HDF5 file has such an appendix. Do not put the
            appendix in the fname string then, e.g. "test_001-P_Ch1-L_Ch2.h5" --> fname="test_001", appendix=True
        :type appendix: bool
        :param which_to_label: Specify which groups from the HDF5 set should be labeled. Possible list members are
            'events', 'testpulses' and 'noise'. In most use cases you will just want the standard argument ['events'].
        :type which_to_label: list of strings
        """

        assert not appendix or channels is not None, 'If you want an automatically appendix you must hand the channels!'

        if appendix:
            if self.nmbr_channels == 2:
                app = '-P_Ch{}-L_Ch{}'.format(*channels)
            else:
                app = ''
                for i, c in enumerate(channels):
                    app += '-{}_Ch{}'.format(i + 1, c)
        else:
            app = ''

        if all([type in self.valid_types for type in which_to_label]):
            self.which_to_label = which_to_label
        else:
            raise ValueError(
                'which_to_label must be a list and contain at least one of events, testpulses, noise.'
            )

        self.fname = fname

        if channels is not None:
            if not len(channels) == self.nmbr_channels:
                raise ValueError('List of channels must vale length {}.'.format(
                    self.nmbr_channels))

        if path == '':
            path = './'
        if path[-1] != '/':
            path = path + '/'
        path_h5 = path + '{}{}.h5'.format(fname, app)
        self.path_h5 = path_h5

        with h5py.File(path_h5, 'r') as f:
            if channels is not None:
                self.channels = channels

            self.nmbrs = {}

            try:
                self.nmbrs['events'] = f['events']['event'].shape[1]
                print('Nmbr triggered events: ', self.nmbrs['events'])
            except KeyError:
                print('No triggered events in h5 file.')

            try:
                self.nmbrs['testpulses'] = f['testpulses']['event'].shape[1]
                print('Nmbr testpulses: ', self.nmbrs['testpulses'])
            except KeyError:
                print('No Testpulses in h5 file.')

            try:
                self.nmbrs['noise'] = f['noise']['event'].shape[1]
                print('Nmbr noise: ', self.nmbrs['noise'])
            except KeyError:
                print('No noise in h5 file.')

        print('HDF5 File loaded.')

    # ------------------------------------------------------------
    # LABELS HANDLING
    # ------------------------------------------------------------

    # Create CSV file for labeling
    def create_labels_csv(self, path: str):
        """
        Create a new CSV file to store the labels.

        The labels are intentionally not included in the HDF5 set right away, to provide a fail-save mechanism in case
        the HDF5 file is re-converted or the this method is accidentally called, overwriting existing labels. Labels are
        usually assigned per hand, making the labels the most time-costly values in your dataset.

        :param path: The path to the file folder where the labels CSV is to be created. A unique naming is automatically
            assigned, e.g. "data/" --> file name "data/labels_bck_001_type.csv".
        :type path: string
        """

        if path == '':
            path = './'
        if path[-1] != '/':
            path = path + '/'
        self.path_csv_labels = path + \
                               'labels_{}_'.format(self.fname)

        try:
            for type in self.which_to_label:
                self.labels[type] = np.zeros(
                    [self.nmbr_channels, self.nmbrs[type]])
                np.savetxt(self.path_csv_labels + type + '.csv',
                           self.labels[type],
                           fmt='%i',
                           delimiter='\n')

        except NameError:
            print('Error! Load a h5 file first.')

    # Load CSV file for labeling
    def load_labels_csv(self, path: str, type: str = 'events'):
        """
        Load a CSV file with labels for the given detector module.

        :param path: the path to the file folder containing the labels CSV file,
            e.g. "data/" --> file name "data/labels_bck_001_type.csv".
        :type path: string
        :param type: The group in the HDF5 corresponding to the labels, either 'events', 'testpulses' or 'noise'.
        :type type: string

        >>> ei.load_labels_csv(path='./')
        Loading Labels from ./labels_test_001_events.csv.
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        if path == '':
            path = './'
        if path[-1] != '/':
            path = path + '/'
        self.path_csv_labels = path + \
                               'labels_{}_'.format(self.fname)

        filename = self.path_csv_labels + type + '.csv'
        print('Loading Labels from {}.'.format(filename))

        labels = np.loadtxt(filename, delimiter='\n')
        labels.resize((self.nmbr_channels, self.nmbrs[type]))
        self.labels[type] = labels

    # Export labels from hdf5 file to CSV file
    def export_labels(self, path: str, type: str = 'events'):
        """
        Save the labels included in the HDF5 file as CSV file.

        You will usually need this option if you have an HDF5 file with included labels, but lost the corresponding CSV
        file. Also, it is recommended to export and store the labels as CSV in on a safe place, e.g. in a Wiki.

        :param path: The path to the file folder containing the labels CSV,
            e.g. "data/" --> file name "data/labels_bck_001_type.csv".
        :type path: string
        :param type: The group in the HDF5 file corresponding to the labels, either 'events', 'testpulses' or 'noise'.
        :type type: string

        >>> ei.export_labels(path='./')
        Labels from HDF5 exported to ./labels_test_001_.
        """

        with h5py.File(self.path_h5, 'r+') as f:

            if not type in self.valid_types:
                raise ValueError('Type should be events, testpulses or noise.')

            if path == '':
                path = './'
            if path[-1] != '/':
                path = path + '/'
            self.path_csv_labels = path + \
                                   'labels_{}_'.format(self.fname)

            # check if hdf5 file has labels
            if not f[type]['labels']:
                print('Load HDF5 File with labels first!')
            else:
                np.savetxt(self.path_csv_labels + type + '.csv',
                           np.array(f[type]['labels']),
                           fmt='%i',
                           delimiter='\n')
                print('Labels from HDF5 exported to {}{}.'.format(
                    self.path_csv_labels, type))

    # ------------------------------------------------------------
    # PREDICTIONS HANDLING
    # ------------------------------------------------------------

    def load_predictions_csv(self,
                             path: str,
                             model: str,
                             type: str = 'events'):
        """
        Load a CSV file with predictions from a machine learning model for the given HDF5 dataset.

        :param path: The path to the file folder containing the CSV file with the predictions,
            e.g. "data/" --> file name "data/<model>_predictions_bck_001_type.csv".
        :type path: string
        :param type: The group in the HDF5 file corresponding to the predictions, either 'events', 'testpulses' or 'noise'.
        :type type: string
        :param model: The name of the model that made the predictions, e.g. "RF" --> Random Forest.
        :type model: string

        >>> ei.load_predictions_csv(path='./', model='RF')
        Loading Predictions from ./RF_predictions_test_001_events.csv.
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        if path == '':
            path = './'
        if path[-1] != '/':
            path = path + '/'
        self.path_csv_predictions = path + \
                                    '{}_predictions_{}_'.format(model, self.fname)

        filename = self.path_csv_predictions + type + '.csv'
        print('Loading Predictions from {}.'.format(filename))

        predictions = np.loadtxt(filename, delimiter='\n')
        predictions.resize((self.nmbr_channels, self.nmbrs[type]))

        # append the predictions
        if type not in self.predictions.keys():
            self.predictions[type] = []
        self.predictions[type].append(predictions)

        # append the model name
        if type not in self.model_names.keys():
            self.model_names[type] = []
        self.model_names[type].append(model)

    def export_predictions(self, path: str, model: str, type: str = 'events'):
        """
        Save the predictions from a machine learning model included in the HDF5 file as CSV file.

        :param path: The path to the file folder containing the predictions CSV,
            e.g. "data/" --> file name "data/<model>_predictions_bck_001_type.csv".
        :type path: string
        :param type: The name of the group in the HDF5 file, either 'events' or 'testpulses' or 'noise'.
        :type type: string
        :param model: The name of the model that made the predictions, e.g. "RF" --> Random Forest.
        :type model: string

        >>> ei.export_predictions(path='./', model='RF')
        RF Predictions from HDF5 exported to RF_predictions_test_001_events.
        """
        with h5py.File(self.path_h5, 'r+') as f:

            if not type in self.valid_types:
                raise ValueError('Type should be events, testpulses or noise.')

            if path == '':
                path = './'
            if path[-1] != '/':
                path = path + '/'
            self.path_csv_predictions = path + \
                                        '{}_predictions_{}_'.format(model, self.fname)

            # check if hdf5 file has labels
            if not f[type]['{}_predictions'.format(model)]:
                print('Load HDF5 File with labels first!')
            else:
                np.savetxt(self.path_csv_predictions + type + '.csv',
                           np.array(f[type]['{}_predictions'.format(model)]),
                           fmt='%i',
                           delimiter='\n')
                print('{} Predictions from HDF5 exported to {}{}.'.format(
                    model, self.path_csv_predictions, type))

    # ------------------------------------------------------------
    # FEATURE HANDLING
    # ------------------------------------------------------------

    # Load OF
    def load_of(self, down: int = 1, group_name_appendix: str = ''):
        """
        Add the optimal transfer function from the HDF5 file.

        :param down: The downsample factor of the optimal transfer function. The data set of the optimumfilter in the HDF5
            set has a consistent name appendix _downX.
        :type down: int
        :param group_name_appendix: A string that is appended to the group name optimumfilter in the HDF5 file. Typically
            this could be _tp for a test pulse optimum filter.
        :type group_name_appendix: str

        This is needed in order to view the filtered event in the labeling or viewing process. For this to work, the
        optimal transfer function must be included in the HDF5 file, e.g. calculated with an instance of the DataHandler
        before (dh.calc_of()).

        >>> ei.load_of()
        Added the optimal transfer function.
        """
        with h5py.File(self.path_h5, 'r') as f:
            if down != 1:
                try:
                    of_real = np.array(
                        f['optimumfilter' + group_name_appendix]['optimumfilter_real_down{}'.format(down)])
                    of_imag = np.array(
                        f['optimumfilter' + group_name_appendix]['optimumfilter_imag_down{}'.format(down)])
                except KeyError:
                    raise KeyError(
                        'Please calculate the OF with according downsampling rate.')
            else:
                of_real = np.array(f['optimumfilter' + group_name_appendix]['optimumfilter_real'])
                of_imag = np.array(f['optimumfilter' + group_name_appendix]['optimumfilter_imag'])

            self.of = of_real + 1j * of_imag
            print('Added the optimal transfer function.')

    def load_sev_par(self, name_appendix='', sample_length=0.04, group_name_appendix: str = ''):
        """
        Add the sev fit parameters from the HDF5 file.

        This is needed in order to view the fitted event in the labeling or viewing process. For this to work, the sev
        fit parameters for every event must be included in the HDF5 file, e.g. calculated with an instance of the
        DataHandler before (dh.apply_sev_fit()).

        :param name_appendix: An appendix to the data set sev_fit_par in the HDF5 set. Typically this is _downX in case
            a downsampling was used for the fit.
        :type name_appendix: string
        :param sample_length: The length of a sample in milliseconds, i.e. 1/sample_frequency.
        :type sample_length: float
        :param group_name_appendix: A string that is appended to the group name stdevent in the HDF5 file. Typically
            this could be _tp for a test pulse standard event.
        :type group_name_appendix: string

        >>> ei.load_sev_par()
        Added the sev fit parameters.
        """

        # save this for loading of the parameters when viewing
        self.name_appendix = name_appendix
        with h5py.File(self.path_h5, 'r') as f:
            sev_par = np.array(f['stdevent' + group_name_appendix]['fitpar'])
            t = (np.arange(0, self.record_length, dtype=float) -
                 self.record_length / 4) * sample_length
            self.fit_models = []
            for c in range(self.nmbr_channels):
                self.fit_models.append(sev_fit_template(pm_par=sev_par[c],
                                                        t=t))

            print('Added the sev fit parameters.')

    def load_arr_par(self, name_appendix='', group_name_appendix: str = '', use_this_array=None):
        """
        Add the array fit parameters from the HDF5 file.

        :param name_appendix: An appendix to the data set arr_fit_par in the HDF5 set.
        :type name_appendix: string
        :param group_name_appendix: A string that is appended to the group name stdevent in the HDF5 file. Typically
            this could be _tp for a test pulse standard event.
        :type group_name_appendix: string
        :param use_this_array:
        :type use_this_array:

        """

        # save this for loading of the parameters when viewing
        self.arr_name_appendix = name_appendix
        if use_this_array is not None:
            self.stdevents = np.array(use_this_array)
        else:
            with h5py.File(self.path_h5, 'r') as f:
                self.stdevents = np.array(f['stdevent' + group_name_appendix]['event'])

        print('Added the array.')

    def load_saturation_par(self):
        """
        Add the saturation fit parameters from the HDF5 file.

        This is needed to show the saturated, fitted events.
        """

        with h5py.File(self.path_h5, 'r') as f:
            self.saturation_pars = np.array(f['saturation']['fitpar'])

        print('Added the saturation fit parameters.')

    def set_threshold(self, threshold: list):
        """
        Set a threshold to show for all channels.

        :param thresholds: The thresholds for all channels.
        :type thresholds: list of floats
        """

        assert len(threshold) == self.nmbr_channels, 'You need to define one threshold for each channel!'
        self.threshold = threshold
        print('Set thresholds to: ', self.threshold)


    # ------------------------------------------------------------
    # LABEL AND VIEWER INTERFACE
    # ------------------------------------------------------------

    def _plot_mp(self,
                 main_par,
                 down: int = 1,
                 color: str = 'r',
                 offset_in_samples: int = 0,
                 xlim: tuple = None,
                 offset_sub: float = 0):
        """
        Function to plot the main parameters, typically accessed by the labeling tool internally.

        :param main_par: list of the 10 main parameters
        :param down: int, the downsample rate
        :param color: string, the color in which the mp are plotted
        :param offset_in_samples: int, an offset parameter from the beginning of the file
        :param xlim: tuple, the lower and upper limit of the x axis in the plot, in sample numbers
        :offset_sub: float, a value that is additionally substracted from the y values
        """
        pulse_height = main_par[0]
        t_zero = main_par[1]
        t_rise = main_par[2]
        t_max = main_par[3]
        t_decaystart = main_par[4]
        t_half = main_par[5]
        t_end = main_par[6]
        offset = main_par[7]

        x_values = [(t_zero - offset_in_samples) / down,
                    (t_rise - offset_in_samples) / down,
                    (t_max - offset_in_samples) / down,
                    (t_decaystart - offset_in_samples) / down,
                    (t_half - offset_in_samples) / down,
                    (t_end - offset_in_samples) / down]

        y_values = [
            offset + 0.1 * pulse_height,
            offset + 0.8 * pulse_height,
            offset + pulse_height,
            offset + 0.9 * pulse_height,
            offset + 0.736 * pulse_height,
            offset + 0.368 * pulse_height
        ]
        x_values = np.array(x_values)
        y_values = np.array(y_values) - offset_sub

        if xlim is not None:
            mask = (x_values >= np.array(xlim[0])) & (x_values <= np.array(xlim[1]))
            plt.scatter(x_values[mask], y_values[mask], color=color, zorder=15)
        else:
            plt.scatter(x_values, y_values, color=color, zorder=15)

    # Access options of label interface
    def _viewer_options(self):
        """
        Prints out all the options that are available in the event viewer/labeling tool
        """
        print('---------- OPTIONS: ----------')
        print('down ... downsample')
        print('der ... show derivative of event')
        print('mp ... show main parameters')
        print('triang ... show triangulation')
        print('of ... show filtered event')
        print('sev ... show fitted standardevent')
        print('arr ... show fitted array')
        print('sat ... show fitted event with saturation')
        print('threshold ... show the trigger threshold')
        print('xlim ... set the x limit')
        print('ylim ... set the y limit')
        print('sub ... subtract offset')
        print('time ... plot time instead of sample index')
        print('window ... include window in of filtering')
        print('q ... quit options menu')

        while True:
            user_input = input('Choose option or q(uit): ')

            # downsample
            if user_input == 'down':
                user_input2 = input('Enter downsample factor (power of 2): ')
                try:
                    down = int(user_input2)
                    if math.log2(down).is_integer():
                        self.down = down
                        self.window_size = int(self.record_length / down)
                        print('Downsample rate set to {}.'.format(self.down))
                    else:
                        print(
                            'Downsample rate has to be integer (power of 2).')
                except ValueError:
                    print('Downsample rate has to be integer (power of 2).')

            # derivative
            elif user_input == 'der':
                self.show_derivative = not self.show_derivative
                self.show_filtered = False
                print('Show derivative set to: ', self.show_derivative)

            # optimum filter
            elif user_input == 'of':
                assert self.of is not None, 'You need to load an optimal filter first!'
                self.show_filtered = not self.show_filtered
                self.show_derivative = False
                print('Show filtered set to: ', self.show_filtered)

            # triangulation
            elif user_input == 'triang':
                self.show_triangulation = not self.show_triangulation
                print('Show triangulation set to: ', self.show_triangulation)

            # main parameters
            elif user_input == 'mp':
                self.show_mp = not self.show_mp
                print('Show Main Parameters set to: ', self.show_mp)

            # sev fit
            elif user_input == 'sev':
                assert self.fit_models is not None, 'You need to load standard event fit parameters first!'
                self.sev = not self.sev
                print('Show SEV fit set to: ', self.sev)

            # arr fit
            elif user_input == 'arr':
                assert self.stdevents is not None, 'You need to load standard event fit parameters first!'
                self.arr = not self.arr
                print('Show SEV fit set to: ', self.arr)

            # saturation
            elif user_input == 'sat':
                assert self.saturation_pars is not None, 'You need to load saturation fit parameters first!'
                self.saturation = not self.saturation
                print('Show saturated fit set to: ', self.sev)

            #
            elif user_input == 'threshold':
                assert self.threshold is not None, 'You need to define the threshold first!'
                self.show_threshold = not self.show_threshold
                print('Show threshold set to: ', self.show_threshold)

            # xlim
            elif user_input == 'xlim':
                user_input2 = input('Set x lower limit: ')
                lb = float(user_input2)
                user_input2 = input('Set x upper limit: ')
                ub = float(user_input2)
                self.xlim = (lb, ub)

            # ylim
            elif user_input == 'ylim':
                user_input2 = input('Set y lower limit: ')
                lb = float(user_input2)
                user_input2 = input('Set y upper limit: ')
                ub = float(user_input2)
                self.ylim = (lb, ub)

            elif user_input == 'sub':
                self.subtract_offset = not self.subtract_offset
                print('Subtract offset set to: ', self.subtract_offset)

            elif user_input == 'time':
                self.show_time = not self.show_time
                print('Plot time instead of index set to: ', self.show_time)

            elif user_input == 'window':
                self.window = not self.window
                print('Include window in OF plot set to: ', self.window)

            # quit
            elif user_input == 'q':
                print('Quit options menu.')
                break

            else:
                print('Please enter a valid option or q to end.')

    # Show specific sample idx from the dataset
    def show(self, idx: int, type: str = 'events'):
        """
        Plots an event of the dataset.

        :param idx: The index of the event that is to show in the hdf5 file.
        :type idx: int
        :param type: The containing group in the HDF5 data set, either 'events', 'testpulses' or 'noise'.
        :type type: string

        >>> ei.show(idx=0)
        Label Phonon: 0.0
        Label Light: 0.0

        .. image:: pics/event.png
        """
        with h5py.File(self.path_h5, 'r+') as f:

            if not type in self.valid_types:
                raise ValueError('Type should be events, testpulses or noise.')

            # get event
            event = np.array(f[type]['event'][:, idx, :])
            appendix = ''

            # downsample first
            if not self.down == 1:
                event = event.reshape((self.nmbr_channels, self.window_size,
                                       self.down))
                event = np.mean(event, axis=2)

            # threshold
            if self.show_threshold:
                threshold = self.threshold + np.mean(
                    event[:, :int(len(event[0]) * self.pre_trigger_region / self.down)],
                    axis=1)

            # optimum filter
            if self.show_filtered:
                try:
                    for c in range(self.nmbr_channels):
                        offset = np.mean(event[c, :int(len(event[c]) * self.pre_trigger_region / self.down)])
                        event[c] = filter_event(event[c] - offset,
                                                self.of[c], window=self.window) + offset
                except ValueError:
                    raise ValueError(console_colors.FAIL + 'ERROR: ' + console_colors.ENDC +
                                     'The downsampling rate of the loaded OF and the Events needs to be the same!')

                appendix = 'Filtered'

            # derivative
            if self.show_derivative:
                event = self.down * \
                        np.diff(event, axis=1, prepend=event[:, 0, np.newaxis])
                appendix = 'Derivative'

            # triangulation
            if self.show_triangulation:
                elements = []
                for i in range(self.nmbr_channels):
                    elements.append(
                        get_elements(event[i], std_thres=self.std_thres[i]))

            # mp
            if self.show_mp:
                main_par = np.array(f[type]['mainpar'][:, idx])

            # sev
            if self.sev:
                try:
                    sev_fit = []
                    fp = f[type]['sev_fit_par{}'.format(self.name_appendix)][:, idx, :]
                    for c in range(self.nmbr_channels):
                        sev_fit.append(self.fit_models[c]._wrap_sec(*fp[c]))
                        if self.saturation:
                            offset = fp[c][2]
                            sev_fit[-1] = scaled_logistic_curve(sev_fit[-1] - offset, *self.saturation_pars[c]) + offset
                except AttributeError:
                    raise AttributeError('No name_appendix attribute, did you load the SEV fit parameters?')

            # arr
            if self.arr:
                try:
                    arr_fit = []
                    fp = f[type]['arr_fit_par{}'.format(self.arr_name_appendix)][:, idx, :]
                    for c in range(self.nmbr_channels):
                        t = (np.arange(0, event.shape[1], 1) - event.shape[1] / 4) * self.down / self.sample_frequency * 1000
                        sev = self.stdevents[c]
                        timebase_ms = 1000 / self.sample_frequency
                        max_shift = np.abs(fp[c, 1]) + 1
                        sample_bounds = int(max_shift / 1000 * self.sample_frequency)
                        arr_fit.append(np.pad(array=template(*fp[c], t[sample_bounds:-sample_bounds],
                                                             sev, timebase_ms, max_shift),
                                              pad_width=sample_bounds, mode='edge'))
                        if self.saturation:
                            offset = fp[c][2]
                            arr_fit[-1] = scaled_logistic_curve(arr_fit[-1] - offset, *self.saturation_pars[c]) + offset
                except AttributeError:
                    raise AttributeError('No arr_name_appendix attribute, did you load the arr fit parameters?')

            # def colors
            if self.nmbr_channels == 1:
                colors = ['blue']
                anti_colors = ['red']
            else:
                colors = ['red' for i in range(self.nmbr_channels - 1)]
                colors.append('blue')
                anti_colors = ['blue' for i in range(self.nmbr_channels - 1)]
                anti_colors.append('red')

            # -------- START PLOTTING --------
            use_cait_style(dpi=self.dpi)
            plt.close()

            for i in range(self.nmbr_channels):

                if self.subtract_offset:
                    offset = np.mean(event[i, :int(len(event[i]) * self.pre_trigger_region / self.down)])
                else:
                    offset = 0

                x = np.arange(0, len(event[i]), 1)

                if self.xlim is not None:
                    mask = x[np.logical_and(x >= np.array(self.xlim[0]),
                                            x <= np.array(self.xlim[1]))]
                else:
                    mask = x

                if self.show_time:
                    x = (np.arange(0, len(event[i]), 1) - len(event[i]) / 4) * self.down / self.sample_frequency

                plt.subplot(self.nmbr_channels, 1, i + 1)
                if not self.show_time:
                    plt.axvline(x=self.window_size / 4, color='grey', alpha=0.6)
                else:
                    plt.axvline(x=0, color='grey', alpha=0.6)
                plt.plot(x[mask],
                         event[i][mask] - offset,
                         label=self.channel_names[i],
                         color=colors[i],
                         zorder=10)
                if self.show_threshold:
                    plt.axhline(y=threshold[i], color='black', alpha=0.8, linewidth=2.5, linestyle='dotted', zorder=30)
                plt.title('Index {}, {} {}'.format(idx,
                                                   self.channel_names[i],
                                                   appendix))

                # triangulation
                if self.show_triangulation:
                    if not self.show_time:
                        plot_S1(event[i], elements[i], color=anti_colors[i], xlim=self.xlim, offset=offset)
                    else:
                        print('Cannot show time and mp.')

                # main parameters
                if self.show_mp:
                    if not self.show_time:
                        self._plot_mp(main_par[i],
                                      color=anti_colors[i],
                                      down=self.down,
                                      xlim=self.xlim,
                                      offset_sub=offset)
                    else:
                        print('Cannot show time and mp.')

                # sev
                if self.sev:
                    plt.plot(x[mask], sev_fit[i][mask] - offset, color='orange', zorder=15)

                # arr
                if self.arr:
                    plt.plot(x[mask], arr_fit[i][mask] - offset, color='c', zorder=15)

                make_grid()
                plt.ylim(self.ylim)

            plt.show(block=False)
            # -------- END PLOTTING --------

            # labels
            if len(self.labels) > 0:
                try:
                    label = self.labels[type][:, idx]

                    for i, nm in enumerate(self.channel_names):
                        print('Label {}: {}'.format(nm, label[i]))

                except NameError:
                    print('No or incorrect Labels.')

            # predictions
            if len(self.predictions) > 0:
                for p_arr in self.predictions[type]:
                    pred = p_arr[:, idx]
                    for i, nm in enumerate(self.channel_names):
                        print('Prediction {}: {}'.format(nm, pred[i]))

            # TPA
            if type == 'testpulses':
                tpa = f['testpulses']['testpulseamplitude']
                if len(tpa.shape) > 1:
                    tpa = f['testpulses']['testpulseamplitude'][:, idx]
                else:
                    tpa = f['testpulses']['testpulseamplitude'][idx]
                print('TPA: {}'.format(tpa))

    def _print_labels(self):
        """
        Prints the labels that are available. The list can be expanded for any given use case.
        """
        print('---------- LABELS: ----------')
        print('0 ... unlabeled')
        print('1 ... Event Pulse')
        print('2 ... Test/Control Pulse')
        print('3 ... Noise')
        print('4 ... Squid Jump')
        print('5 ... Spike')
        print('6 ... Early or late Trigger')
        print('7 ... Pile Up')
        print('8 ... Carrier Event')
        print('9 ... Strongly Saturated Event Pulse')
        print('10 ... Strongly Saturated Test/Control Pulse')
        print('11 ... Decaying Baseline')
        print('12 ... Temperature Rise')
        print('13 ... Stick Event')
        print('14 ... Square Waves')
        print('15 ... Human Disturbance')
        print('16 ... Large Sawtooth')
        print('17 ... Cosinus Tail')
        print('18 ... Light only Event')
        print('19 ... Ring & Light Event')
        print('20 ... Sharp Light Event')
        print('99 ... unknown/other')

    def _ask_for_options(self, user_input):
        if user_input == 'q':
            return -1
        elif user_input == 'b':
            return -2
        elif user_input == 'n':
            return -3
        elif user_input == 'o':
            return -4
        elif user_input == 'i':
            return -5
        elif user_input == 'p':
            return -6
        else:
            print(
                'Enter q end, b back, n next, o options, i idx, p for (de)activate label list\n')

    def _ask_for_label(self, idx: int, which: str = 'phonon'):
        """
        Takes and processes an user input to the viewer/labeling tool.

        :param idx: int, the index of the event that is to label in the h5 file
        :param which: string, the naming of the channel, e.g. phonon/light
        :return: int > 0 or option code (int < 0) if the user input was one of the option flag
        """
        print(
            'Assign label for event idx: {} {} (q end, b back, n next, o options, i idx, p for (de)activate label list)\n'.format(
                idx, which))

        while True:
            user_input = input('{}: '.format(which))
            try:
                label = int(user_input)
                if label > 0:
                    return label
                else:
                    print(
                        'Enter Integer > 0 or q end, b back, n next, o options, i idx, p for (de)activate label list')
            except ValueError:
                return self._ask_for_options(user_input)

    def _ask_for_idx(self, length: int):
        """
        Gets an index from the user to which we want to jump.

        :param length: int, maximal index the user may put
        :return: int, the index the used put
        :raises ValueError if the user input was not a valid index
        """
        while True:
            user_input = input('Jump to which index? ')
            try:
                idx = int(user_input)
                if (idx < 0) or (idx >= length):
                    raise ValueError('This is not a valid index!')
                else:
                    print('Jumping to index ', idx)
                    return idx
            except ValueError:
                print('Enter valid index!')

    def start(self,
              start_from_idx: int = 0,
              print_label_list: bool = True,
              label_only_class: int = None,
              label_only_prediction: int = None,
              model: str = None,
              viewer_mode: bool = False):
        """
        Starts the label/view interface.

        The user gets the events shown and is asked for labels. There are viewer options available:

        - n … next sample
        - b … previous sample
        - idx ... user is asked for an index to jump to
        - q … quit
        - o … show set options and ask for changes, options are down - der - mp - predRF - predLSTM - triang - of - … - q

        The options in the options menu change the displayed event:

        - down ... Event is downsampled before plotting, this smoothes the noise.
        - der ... The derivative of the event is shown.
        - mp ... The main parameters are visualized as scatter points in the plot.
        - triang ... A triangulation of the event is shown. This feature is experimental and not supported any longer.
        - of ... The filtered event is shown.
        - q ... Quit the menu and go back to the labeling interface.

        There are more options, explore them in the options menu!

        :param start_from_idx: An event index to start labeling from.
        :type start_from_idx: int
        :param print_label_list: If set to true, the list of the labels is printed together when the user is asked for
            labels.
        :type print_label_list: bool
        :param label_only_class: If set only events of this class will be shown in the labeling/viewing process.
        :type label_only_class: int
        :param label_only_prediction: If set only events of this prediction will be shown in the labeling/viewing process.
        :type label_only_prediction: int
        :param model: The naming of the model that made the predictions, e.g. 'RF' for Random Forest
        :type model: string
        :param viewer_mode: Activates viewer mode. Labelling is not possible while in viewer mode.
        :type viewer_mode: bool
        """
        if label_only_class:
            print('Start labeling from idx {}, label only class {}.'.format(
                start_from_idx, label_only_class))
            # label_all_classes = False
        elif label_only_prediction:
            print('Start labeling from idx {}, label only prediction {}.'.format(
                start_from_idx, label_only_prediction))
            # label_all_classes = False
        else:
            print('Start labeling from idx {}.'.format(start_from_idx))
            # label_all_classes = True
        if not viewer_mode:
            try:
                print('Labels autosave to {}.'.format(self.path_csv_labels))
            except:
                while True:
                    print("No Labels file! Do you want to start in viewer mode (y/n)?")
                    viewer_mode = input()
                    if viewer_mode.lower() == 'y':
                        viewer_mode = True
                        print('You have selected viewer mode!')
                        print(
                            'Navigate through events by pressing b back or n next. All other options are also available.')
                        break
                    elif viewer_mode.lower() == 'n':
                        raise AttributeError('Load or create labels file first!')
                    else:
                        print('Please enter a valid input! Either y or n')

        for type in self.which_to_label:

            idx = np.copy(start_from_idx)

            while idx < self.nmbrs[type]:
                if label_only_class is not None:
                    class_condition = (label_only_class == self.labels[type][:, idx]).any()
                else:
                    class_condition = True
                if label_only_prediction is not None:
                    preds = self.predictions[type][self.model_names[type].index(model)][:, idx]
                    prediction_condition = (label_only_prediction == preds).any()
                    del preds
                else:
                    prediction_condition = True

                if class_condition and prediction_condition:  # or label_all_classes:

                    if print_label_list and not viewer_mode:
                        self._print_labels()
                    self.show(idx, type)

                    for i, channel in enumerate(self.channel_names):
                        if not viewer_mode:
                            user_input = self._ask_for_label(idx, channel)
                        else:
                            user_input = input(
                                'Enter q end, b back, n next, o options, i idx, p for (de)activate label list')
                            user_input = self._ask_for_options(user_input)

                        if user_input == -1:
                            print('End labeling.')
                            idx = self.nmbrs[type]
                            break
                        elif user_input == -2:
                            print('Rolling back to previous.')
                            idx -= 2
                            break
                        elif user_input == -3:
                            print('Skipping this label.')
                            break
                        elif user_input == -4:
                            self._viewer_options()
                            idx -= 1
                            break
                        elif user_input == -5:
                            idx = self._ask_for_idx(self.nmbrs[type]) - 1
                            break
                        elif user_input == -6:
                            print_label_list = not print_label_list
                            idx -= 1
                            break
                        elif not viewer_mode:
                            self.labels[type][i, idx] = user_input
                            np.savetxt(self.path_csv_labels + type + '.csv',
                                       self.labels[type],
                                       fmt='%i',
                                       delimiter='\n')
                        else:
                            print('Only option keys are valid!')
                            pass

                idx += 1
