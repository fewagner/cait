# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
from .features._fem import get_elements, plot_S1
from .filter._of import filter_event
from .fit._templates import sev_fit_template

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class EventInterface:
    """
    A class for the viewing and labeling of Events from HDF5 data sets.
    """

    def __init__(self, module, run, record_length,
                 sample_frequency=25000, nmbr_channels=2, down=1):
        """
        Provide general information about the detector for a new instance of the class.

        :param module: string, the naming of the detector module
        :param run: int, the number of the run from which the measurement comes
        :param record_length: int, the number of samples in a record window
        :param sample_frequency: int, the record frequency of the measurement
        :param nmbr_channels: int, the number of channels of the detector modules
        :param down: int, the downsample rate for viewing the events
        """

        if nmbr_channels not in [2, 3]:
            raise ValueError("Channel Number must be 2 or 3!")
        else:
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
        self.labels = {}
        self.predictions = {}
        self.model_names = {}
        self.valid_types = ['events', 'testpulses', 'noise']
        if self.nmbr_channels == 2:
            self.channel_names = ['Phonon', 'Light']
        elif self.nmbr_channels == 3:
            self.channel_names = ['Channel 1', 'Channel 2', 'Channel 3']
        self.xlim = None
        self.ylim = None

        print('Event Interface Instance created.')

    # ------------------------------------------------------------
    # INCLUDE THE DATA
    # ------------------------------------------------------------

    # Load in the hdf5 dataset
    def load_bck(self, path, bck_nmbr, channels,
                 bck_naming='bck',
                 appendix=True,
                 which_to_label=['events']):
        """
        Load a hdf5 dataset to the instance

        :param path: string, path to the file folder; e.g. "data/" --> filepath "data/runXY_MODULE/bck_nmbr-[appendix].h5"
        :param bck_nmbr: string, the appended number of the file; e.g. "bck_001" --> "001"
        :param channels: list of strings, the numbers of the channels that are included in the bck file
        :param bck_naming: string, the file naming, e.g. bck, cal, blue, ....
        :param appendix: bool, if True the appendix generated from the gen_h5_from_rdt function is appended to the name
        :param which_to_label: list of strings, possible members are events, testpulses, noise
        :return: -
        """

        if appendix:
            if self.nmbr_channels == 2:
                app = '-P_Ch{}-L_Ch{}'.format(*channels)
            elif self.nmbr_channels == 3:
                app = '-1_Ch{}-2_Ch{}-3_Ch{}'.format(*channels)
        else:
            app = ''

        if all([type in self.valid_types for type in which_to_label]):
            self.which_to_label = which_to_label
        else:
            raise ValueError(
                'which_to_label must be a list and contain at least one of events, testpulses, noise.')

        self.bck_naming = bck_naming
        self.bck_nmbr = bck_nmbr

        if not len(channels) == self.nmbr_channels:
            raise ValueError(
                'List of channels must vale length {}.'.format(self.nmbr_channels))

        path_h5 = path + 'run{}_{}/{}_{}{}.h5'.format(self.run,
                                                                  self.module,
                                                                  bck_naming,
                                                                  bck_nmbr,
                                                                  app)

        self.f = h5py.File(path_h5, 'r')
        self.channels = channels

        self.nmbrs = {}

        try:
            self.nmbrs['events'] = len(self.f['events']['event'][0])
            print('Nmbr triggered events: ', self.nmbrs['events'])
        except KeyError:
            print('No triggered events in h5 file.')

        try:
            self.nmbrs['testpulses'] = len(self.f['testpulses']['event'][0])
            print('Nmbr testpulses: ', self.nmbrs['testpulses'])
        except KeyError:
            print('No Testpulses in h5 file.')

        try:
            self.nmbrs['noise'] = len(self.f['noise']['event'][0])
            print('Nmbr noise: ', self.nmbrs['noise'])
        except KeyError:
            print('No noise in h5 file.')

        print('Bck File loaded.')

    # ------------------------------------------------------------
    # LABELS HANDLING
    # ------------------------------------------------------------

    # Create CSV file for labeling
    def create_labels_csv(self, path):
        """
        Create a new CSV file to store the labels

        :param path: string, the path to the file folder,
            e.g. "data/" --> file name "data/runXY_MODULE/labels_bck_001_type.csv"
        :return: -
        """

        self.path_csv_labels = path + \
            'run{}_{}/labels_{}_{}_'.format(self.run,
                                            self.module, self.bck_naming, self.bck_nmbr)

        try:
            for type in self.which_to_label:
                self.labels[type] = np.zeros(
                    [self.nmbr_channels, self.nmbrs[type]])
                np.savetxt(self.path_csv_labels + type + '.csv',
                           self.labels[type], delimiter='\n')

        except NameError:
            print('Error! Load a bck file first.')

    # Load CSV file for labeling
    def load_labels_csv(self, path, type):
        """
        Load a csv file with labels

        :param path: string, the path to the file folder,
            e.g. "data/" --> file name "data/runXY_MODULE/labels_bck_001_type.csv"
        :param type: string, either events, testpulses or noise
        :return: -
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        self.path_csv_labels = path + \
            'run{}_{}/labels_{}_{}_'.format(self.run,
                                            self.module, self.bck_naming, self.bck_nmbr)

        filename = self.path_csv_labels + type + '.csv'
        print('Loading Labels from {}.'.format(filename))

        labels = np.loadtxt(filename, delimiter='\n')
        labels.resize((self.nmbr_channels, self.nmbrs[type]))
        self.labels[type] = labels

    # Export labels from hdf5 file to CSV file
    def export_labels(self, path, type):
        """
        Save the labels included in the HDF5 file as CSV file

        :param path: string, the path to the file folder,
            e.g. "data/" --> file name "data/runXY_MODULE/labels_bck_001_type.csv"
        :param type: string, either events or testpulses or noise
        :return: -
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        self.path_csv_labels = path + \
            'run{}_{}/labels_{}_{}_'.format(self.run,
                                            self.module, self.bck_naming, self.bck_nmbr)

        # check if hdf5 file has labels
        if not self.f[type]['labels']:
            print('Load HDF5 File with labels first!')
        else:
            np.savetxt(self.path_csv_labels + type + '.csv',
                       np.array(self.f[type]['labels']), delimiter='\n')
            print('Labels from HDF5 exported to {}.'.format(self.path_csv_labels))

    # ------------------------------------------------------------
    # PREDICTIONS HANDLING
    # ------------------------------------------------------------

    def load_predictions_csv(self, path, type, model):
        """
        Load a csv file with predictions

        :param path: string, the path to the file folder,
            e.g. "data/" --> file name "data/runXY_MODULE/<model>_predictions_bck_001_type.csv"
        :param type: string, either events, testpulses or noise
        :param model: string, the name of the model that made the predictions, e.g. "RF" --> Random Forest
        :return: -
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        self.path_csv_predictions = path + \
            'run{}_{}/{}_predictions_{}_{}_'.format(self.run,
                                            self.module, model, self.bck_naming, self.bck_nmbr)

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


    def export_predictions(self, path, type, model):
        """
        Save the predictions included in the HDF5 file as CSV file

        :param path: string, the path to the file folder,
            e.g. "data/" --> file name "data/runXY_MODULE/<model>_predictions_bck_001_type.csv"
        :param type: string, either events or testpulses or noise
        :param model: string, the name of the model that made the predictions, e.g. "RF" --> Random Forest
        :return: -
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        self.path_csv_predictions = path + \
            'run{}_{}/{}_predictions_{}_{}_'.format(self.run,
                                            self.module, model, self.bck_naming, self.bck_nmbr)

        # check if hdf5 file has labels
        if not self.f[type]['{}_predictions'.format(model)]:
            print('Load HDF5 File with labels first!')
        else:
            np.savetxt(self.path_csv_predictions + type + '.csv',
                       np.array(self.f[type]['{}_predictions'.format(model)]), delimiter='\n')
            print('{} Predictions from HDF5 exported to {}.'.format(model, self.path_csv_predictions))

    # ------------------------------------------------------------
    # FEATURE HANDLING
    # ------------------------------------------------------------

    # Load OF
    def load_of(self):
        """
        Add the optimal transfer function from the HDF5 file
        """
        of_real = np.array(self.f['optimumfilter']['optimumfilter_real'])
        of_imag = np.array(self.f['optimumfilter']['optimumfilter_imag'])
        self.of = of_real + 1j*of_imag
        print('Added the optimal transfer function.')

    def load_sev_par(self, sample_length=0.04):
        """
        Add the sev fit parameters from the HDF5 file
        """
        sev_par = np.array(self.f['stdevent']['fitpar'])
        t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length
        self.fit_models = []
        for c in range(self.nmbr_channels):
            self.fit_models.append(sev_fit_template(pm_par=sev_par[c], t=t))

        print('Added the sev fit parameters.')


    # ------------------------------------------------------------
    # LABEL AND VIEWER INTERFACE
    # ------------------------------------------------------------

    def _plot_mp(self, main_par, down=1, color='r', offset_in_samples=0):
        """
        Function to plot the main parameters, typically accessed by the labeling tool internally

        :param main_par: list of the 10 main parameters
        :param down: int, the downsample rate
        :param color: string, the color in which the mp are plotted
        :param offset_in_samples: int, an offset parameter from the beginning of the file
        :return: -
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

        y_values = [offset + 0.1 * pulse_height,
                    offset + 0.8 * pulse_height,
                    offset + pulse_height,
                    offset + 0.9 * pulse_height,
                    offset + 0.736 * pulse_height,
                    offset + 0.368 * pulse_height]

        plt.scatter(x_values, y_values, color=color)

    # Access options of label interface
    def viewer_options(self):
        """
        Prints out all the options that are available in the event viewer/labeling tool

        :return: -
        """
        print('---------- OPTIONS: ----------')
        print('down ... downsample')
        print('der ... show derivative of event')
        print('mp ... show main parameters')
        print('triang ... show triangulation')
        print('of ... show filtered event')
        print('sev ... show fitted standardevent')
        print('xlim ... set the x limit')
        print('ylim ... set the y limit')
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
                        print('Downsample rate has to be integer (power of 2).')
                except ValueError:
                    print('Downsample rate has to be integer (power of 2).')

            # derivative
            elif user_input == 'der':
                self.show_derivative = not self.show_derivative
                self.show_filtered = False
                print('Show derivative set to: ', self.show_derivative)

            # optimum filter
            elif user_input == 'of':
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
                self.sev = not self.sev
                print('Show SEV fit set to: ', self.sev)

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

            # quit
            elif user_input == 'q':
                print('Quit options menu.')
                break

            else:
                print('Please enter a valid option or q to end.')

    # Show specific sample idx from the dataset
    def show(self, idx, type):
        """
        Plots an event

        :param idx: the index of the event that is to show in the hdf5 file
        :param type: string, either events, testpulses or noise
        :return: -
        """

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        # get event
        event = np.array(self.f[type]['event'][:, idx, :])
        appendix = ''

        # optimum filter
        if self.show_filtered:
            for c in range(self.nmbr_channels):
                offset = np.mean(event[c, :int(len(event[c])/8)])
                event[c] = filter_event(event[c] - offset, self.of[c]) + offset
            appendix = 'Filtered'

        # downsample
        if not self.down == 1:
            event = event.reshape(self.nmbr_channels,
                                  self.window_size, self.down)
            event = np.mean(event, axis=2)

        # derivative
        if self.show_derivative:
            event = self.down * \
                np.diff(event, axis=1, prepend=event[:, 0, np.newaxis])
            appendix = 'Derivative'

        # triangulation
        if self.show_triangulation:
            elements = []
            for i in range(self.nmbr_channels):
                elements.append(get_elements(
                    event[i], std_thres=self.std_thres[i]))

        # mp
        if self.show_mp:
            main_par = np.array(self.f[type]['mainpar'][:, idx])

        # sev
        if self.sev:
            sev_fit = []
            fp = self.f['events']['sev_fit_par'][:, idx, :]
            for c in range(self.nmbr_channels):
                offset = np.mean(event[c, :int(len(event[c]) / 8)])
                sev_fit.append(self.fit_models[c].sec(*fp[c]) + offset)

        # def colors
        if self.nmbr_channels == 2:
            colors = ['blue', 'red']
            anti_colors = ['red', 'blue']
        elif self.nmbr_channels == 3:
            colors = ['red', 'red', 'blue']
            anti_colors = ['blue', 'blue', 'red']

        # -------- START PLOTTING --------
        plt.close()

        for i in range(self.nmbr_channels):

            plt.subplot(self.nmbr_channels, 1, i + 1)
            plt.axvline(x=self.window_size / 4, color='grey', alpha=0.6)
            plt.plot(event[i], label=self.channel_names[i], color=colors[i])
            plt.title('{} {} + {} {}'.format(type, idx,
                                             self.channel_names[i], appendix))

            # triangulation
            if self.show_triangulation:
                plot_S1(event[i], elements[i], color=anti_colors[i])

            # main parameters
            if self.show_mp:
                self._plot_mp(
                    main_par[i], color=anti_colors[i], down=self.down)

            # sev
            if self.sev:
                plt.plot(sev_fit[i], color='orange')

            plt.xlim(self.xlim)
            plt.ylim(self.ylim)

        plt.show(block=False)
        # -------- END PLOTTING --------

        # labels
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
            tpa = self.f['testpulses']['testpulseamplitude'][idx]
            print('TPA: {}'.format(tpa))


    def _print_labels(self):
        """
        Prints the labels that are available

        :return: -
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
        print('99 ... unknown/other')

    def _ask_for_label(self, idx, which='phonon'):
        """
        Takes and processes an user input to the viewer/labeling tool

        :param idx: int, the index of the event that is to label in the h5 file
        :param which: string, the naming of the channel, e.g. phonon/light
        :return: int > 0 or option code (int < 0) if the user input was one of the option flag
        """
        print('Assign label for event idx: {} {} (q end, b back, n next, o options, i idx)'.format(
            idx, which))

        while True:
            user_input = input('{}: '.format(which))
            try:
                label = int(user_input)
                if label > 0:
                    return label
                else:
                    print(
                        'Enter Integer > 0 or q end, b back, n next, o options, i idx')
            except ValueError:
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
                else:
                    print(
                        'Enter Integer > 0 or q end, b back, n next, o options, i idx')

    def _ask_for_idx(self, length):
        """
        Gets an index from the user to which we want to jump

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


    def start_labeling(self,
                       start_from_idx,
                       label_only_class=None,
                       label_only_prediction=None,
                       model=None):
        """
        Starts the label/view interface
        The user gets the events shown and is asked for labels. There are viewer options available:
        There are four options: n, b, i -idx, o, q
        n … next sample
        b … previous sample
        q … quit
        o … show set options and ask for changes, options are down - der - mp - predRF - predLSTM - triang - of - … - q

        :param start_from_idx: int, an index to start labeling from
        :param label_only_class: int, if set only events of this class will be shown
        :param label_only_prediction: int, if set only events of this prediction will be shown
        :param model: string, the naming of the model that made the predictions
        :return: -
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

        try:
            print('Labels autosave to {}.'.format(self.path_csv_labels))
        except AttributeError:
            print('Load or create labels file first!')

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

                if class_condition and prediction_condition: # or label_all_classes:

                    self._print_labels()
                    self.show(idx, type)

                    for i, channel in enumerate(self.channel_names):
                        user_input = self._ask_for_label(idx, channel)

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
                            self.viewer_options()
                            idx -= 1
                            break
                        elif user_input == -5:
                            idx = self._ask_for_idx(self.nmbrs[type]) - 1
                            break
                        else:
                            self.labels[type][i, idx] = user_input
                            np.savetxt(self.path_csv_labels + type + '.csv',
                                       self.labels[type], delimiter='\n')

                idx += 1
