"""
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from .features._fem import get_elements, plot_S1
from .features._ts_feat import calc_ts_features

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class EventInterface:

    def __init__(self, module, run, record_length,
                 sample_frequency=25000, nmbr_channels=2, down=1):

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
        self.rf_pred = False
        self.lstm_pred = False
        self.only_wrong = False
        self.show_filtered = False
        self.valid_types = ['events', 'testpulses', 'noise']
        if self.nmbr_channels == 2:
            self.channel_names = ['Phonon', 'Light']
        elif self.nmbr_channels == 3:
            self.channel_names = ['Channel 1', 'Channel 2', 'Channel 3']

        print('Event Interface Instance created.')

    # Load in the hdf5 dataset
    def load_bck(self, path, bck_nmbr, channels,
                 bck_naming='bck',
                 which_to_label=['events']):

        if all([type in self.valid_types for type in which_to_label]):
            self.which_to_label = which_to_label
        else:
            raise ValueError('which_to_label must be a list and contain at least one of events, testpulses, noise.')

        self.bck_naming = bck_naming
        self.bck_nmbr = bck_nmbr

        if not len(channels) == self.nmbr_channels:
            raise ValueError('List of channels must vale length {}.'.format(self.nmbr_channels))

        if len(channels) == 2:
            path_h5 = path + 'run{}_{}/{}_{}-P_Ch{}-L_Ch{}.h5'.format(self.run,
                                                                      self.module,
                                                                      bck_naming,
                                                                      bck_nmbr,
                                                                      *channels)
        elif len(channels) == 3:
            path_h5 = path + 'run{}_{}/{}_{}-1_Ch{}-2_Ch{}-3_Ch{}.h5'.format(self.run,
                                                                             self.module,
                                                                             bck_naming,
                                                                             bck_nmbr,
                                                                             *channels)
        else:
            print('Need 2 or 3 channels!')
            return

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

    # Create CSV file for labeling
    def create_labels_csv(self, path):

        self.path_csv = path + 'run{}_{}/labels_{}_{}_'.format(self.run, self.module, self.bck_naming, self.bck_nmbr)

        self.labels = {}

        try:
            for type in self.which_to_label:
                self.labels[type] = np.zeros([self.nmbr_channels, self.nmbrs[type]])
                np.savetxt(self.path_csv + type + '.csv', self.labels[type], delimiter='\n')

        except NameError:
            print('Error! Load a bck file first.')

    # Load CSV file for labeling
    def load_labels_csv(self, path, type):

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        self.path_csv = path + 'run{}_{}/labels_{}_{}_'.format(self.run, self.module, self.bck_naming, self.bck_nmbr)

        filename = self.path_csv + type + '.csv'
        print('Loading Labels from {}.'.format(filename))

        labels = np.loadtxt(filename, delimiter='\n')
        self.labels = {}

        labels.resize((self.nmbr_channels, self.nmbrs[type]))
        self.labels[type] = labels

    # Export labels from hdf5 file to CSV file
    def export_labels(self, path, type):

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        self.path_csv = path + 'run{}_{}/labels_{}_{}_'.format(self.run, self.module, self.bck_naming, self.bck_nmbr)

        # check if hdf5 file has labels
        if not self.f[type]['labels']:
            print('Load HDF5 File with labels first!')
        else:
            np.savetxt(self.path_csv + type + '.csv', np.array(self.f[type]['labels']), delimiter='\n')
            print('Labels from HDF5 imported to {}.'.format(self.path_csv))

    # Load RF model(s), also define downsample rate
    # TODO test this function
    def load_rf(self, path):
        path_model = '{}/rf_{}_{}'.format(path, self.run, self.module)
        rf_model = pickle.load(open(path_model, 'rb'))
        if rf_model.down == self.down:
            self.rf_model = rf_model
            print('RF model loaded from {}.'.format(path_model))
        else:
            raise ValueError('Downsample rate must match.')

    # Load LSTM model(s), also define downsample rate
    # TODO test this function
    def load_lstm(self, path):
        path_model = '{}/lstm_{}_{}'.format(path, self.run, self.module)
        lstm_model = pickle.load(open(path_model, 'rb'))
        if lstm_model.down == self.down:
            self.lstm_model = lstm_model
            print('LSTM model loaded from {}.'.format(path_model))
        else:
            raise ValueError('Downsample rate must match.')

    # Load NPS, also define downsample rate
    # TODO test this function
    def load_nps(self, path, length_event):
        raise NotImplementedError('Not implemented!')

    # Load Stdevent, also define downsample rate
    # TODO test this function
    def load_sev(self, path, length_event):
        raise NotImplementedError('Not implemented!')

    # Load OF, also define downsample rate
    # TODO test this function
    def load_of(self, path, length_event):
        raise NotImplementedError('Not implemented!')

    # Calculate Features with library
    def calculate_features(self, type, scaler=None):

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        try:
            self.features
        except AttributeError:
            self.features = {}

        try:
            events = np.array(self.f[type]['event'])
            mainpar = np.array(self.f[type]['mainpar'])
        except NameError:
            print('Load according Bck File first!')
            return

        self.features[type] = calc_ts_features(events=events,
                                               mainpar=mainpar,
                                               nmbr_channels=self.nmbr_channels,
                                               nmbrs=self.nmbrs[type],
                                               down=self.down,
                                               sample_frequency=self.sample_frequency,
                                               scaler=scaler)
        print('Features calculated.')

    # Save Features with library
    def save_features(self, path):
        try:
            path_features = '{}run{}_{}/features_{}_{}'.format(path, self.run, self.module, self.bck_naming, self.bck_nmbr)
            pickle.dump(self.features, open(path_features, 'wb'))
            print('Saved Features to {}.'.format(path_features))
        except AttributeError:
            print('Calculate or Load Features first!')

    # Load Features with library
    def load_features(self, path):
        path_features = '{}run{}_{}/features'.format(path, self.run, self.module)
        self.features = pickle.load(open(path_features, 'rb'))
        print('Loaded Features from {}.'.format(path_features))

    def _plot_mp(self, main_par, down=1, color='r', offset_in_samples=0):
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
        print('---------- OPTIONS: ----------')
        print('down ... downsample')
        print('der ... show derivative of event')
        print('mp ... show main parameters')
        print('rf ... show prediction of RF')
        print('lstm ... show prediction of LSTM')
        print('triang ... show triangulation')
        print('of ... show filtered event')
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

            # random forest
            elif user_input == 'rf':
                self.rf_pred = not self.rf_pred
                print('Show RF prediction set to: ', self.rf_pred)

            # lstm
            elif user_input == 'lstm':
                self.lstm_pred = not self.lstm_pred
                print('Show LSTM prediction set to: ', self.lstm_pred)

            # quit
            elif user_input == 'q':
                print('Quit options menu.')
                break

            else:
                print('Please enter a valid option or q to end.')

    # Show specific sample idx from the dataset
    def show(self, idx, type):

        if not type in self.valid_types:
            raise ValueError('Type should be events, testpulses or noise.')

        # get event
        event = np.array(self.f[type]['event'][:, idx, :])
        appendix = ''

        # downsample
        if not self.down == 1:
            event = event.reshape(self.nmbr_channels, self.window_size, self.down)
            event = np.mean(event, axis=2)

        # derivative
        if self.show_derivative:
            event = self.down * np.diff(event, axis=1, prepend=event[:, 0, np.newaxis])
            appendix = 'Derivative'

        # optimum filter
        elif self.show_filtered:
            raise NotImplementedError('Not implemented!')
            appendix = 'Filtered'

        # triangulation
        if self.show_triangulation:
            elements = []
            for i in range(self.nmbr_channels):
                elements.append(get_elements(event[i], std_thres=self.std_thres[i]))

        # mp
        if self.show_mp:
            main_par = np.array(self.f[type]['mainpar'][:, idx])

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
            plt.title('{} {} + {} {}'.format(type, idx, self.channel_names[i], appendix))

            # triangulation
            if self.show_triangulation:
                plot_S1(event[i], elements[i], color=anti_colors[i])

            # main parameters
            if self.show_mp:
                self._plot_mp(main_par[i], color=anti_colors[i], down=self.down)

        plt.show(block=False)
        # -------- END PLOTTING --------

        # labels
        try:
            label = self.labels[type][:, idx]

            for i, nm in enumerate(self.channel_names):
                print('Label {}: {}'.format(nm, label[i]))

        except NameError:
            print('No or incorrect Labels.')

        # rf
        if self.rf_pred:
            try:
                # what about features??
                print('RF not implmented.')
            except AttributeError:
                print('No RF.')

        # lstm
        if self.lstm_pred:
            try:
                self.lstm_model(event)
            except AttributeError:
                print('No LSTM.')

    def _print_labels(self):
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
        print('14 ... Sawtooth Cycle')
        print('99 ... unknown/other')

    def _ask_for_label(self, idx, which='phonon'):
        print('Assign label for event idx: {} {} (q end, b back, n next, o options, i idx)'.format(idx, which))

        while True:
            user_input = input('{}: '.format(which))
            try:
                label = int(user_input)
                if label > 0:
                    return label
                else:
                    print('Enter Integer > 0 or q end, b back, n next, o options, i idx')
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
                    print('Enter Integer > 0 or q end, b back, n next, o options, i idx')

    def _ask_for_idx(self, length):
        while True:
            user_input = input('Jump to which index? ')
            try:
                idx = int(user_input)
                if (idx < 0) or (idx >= length):
                    raise ValueError
                else:
                    print('Jumping to index ', idx)
                    return idx
            except ValueError:
                print('Enter valid index!')

    # Start labeling from idx and choose if only specific labels or only wrong predictions are shown
    # There are four options: n, b, i -idx, o, q
    # n … next sample
    # b … previous sample
    # q … quit
    # o … show set options and ask for changes, options are down - der - mp - predRF - predLSTM - triang - of - … - q
    def start_labeling(self, start_from_idx, label_only_class=None):
        if label_only_class:
            print('Start labeling from idx {}, label only class {}.'.format(start_from_idx, label_only_class))
        else:
            print('Start labeling from idx {}.'.format(start_from_idx))
            label_all_classes = True

        try:
            print('Labels autosave to {}.'.format(self.path_csv))
        except AttributeError:
            print('Load or create labels file first!')

        for type in self.which_to_label:

            idx = np.copy(start_from_idx)

            while idx < self.nmbrs[type]:
                if (label_only_class == self.labels[type][:, idx]).any() or (label_all_classes):

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
                            np.savetxt(self.path_csv + type + '.csv', self.labels[type], delimiter='\n')

                idx += 1
