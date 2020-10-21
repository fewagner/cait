"""
"""

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import os
import numpy as np
import h5py
from multiprocessing import Pool
import struct
import matplotlib.pyplot as plt
from .data._gen_h5 import gen_dataset_from_rdt
from .features._mp import calc_main_parameters
from .fit._sev import generate_standard_event
from .filter._of import optimal_transfer_function


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler:

    def __init__(self, run, module, channels, record_length):
        # ask user things like which detector working on etc
        if len(channels) != 2:
            raise NotImplementedError('Only for 2 channels implemented.')
        self.run = run
        self.module = module
        self.record_length = record_length
        self.nmbr_channels = len(channels)
        self.channels = channels

        print('DataHandler Instance created.')

    # Checkout RDT File if Channel exists etc
    def checkout_rdt(self, path_rdt, read_events=-1, tpa_list=[0], dvm_channels=0):
        # initialize dvm and event arrays
        dvm = np.zeros(dvm_channels, dtype=float)
        event = np.zeros(self.record_length, dtype=np.short)

        detectornumbers = np.empty([read_events])

        with open(path_rdt, "rb") as f:
            for nmbr_event in range(read_events):

                # read all header infos of the event
                detector_nmbr = struct.unpack('i', f.read(4))[0]
                coincide_pulses = struct.unpack('i', f.read(4))[0]
                trig_count = struct.unpack('i', f.read(4))[0]
                trig_delay = struct.unpack('i', f.read(4))[0]
                abs_time_s = struct.unpack('i', f.read(4))[0]
                abs_time_mus = struct.unpack('i', f.read(4))[0]
                delay_ch_tp = struct.unpack('i', f.read(4))[0]
                time_low = struct.unpack('I', f.read(4))[0]  # 'L'
                time_high = struct.unpack('I', f.read(4))[0]  # 'L'
                qcd_events = struct.unpack('I', f.read(4))[0]  # 'L'
                hours = struct.unpack('f', f.read(4))[0]  # 'f'
                dead_time = struct.unpack('f', f.read(4))[0]  # 'f'
                test_pulse_amplitude = struct.unpack('f', f.read(4))[0]  # 'f'
                dac_output = struct.unpack('f', f.read(4))[0]  # 'f'

                # read the dvm channels
                for i in range(dvm_channels):
                    dvm[i] = struct.unpack('f', f.read(4))[0]  # 'f'

                # read the recorded event
                for i in range(self.record_length):
                    event[i] = struct.unpack('h', f.read(2))[0]  # 'h'

                # print all headerinfos

                if (test_pulse_amplitude in tpa_list):
                    print(
                        '#############################################################')
                    print('EVENT NUMBER: ', nmbr_event)

                    print('detector number (starting at 0): ', detector_nmbr)

                    # print('number of coincident pulses in digitizer module: ', coincide_pulses)
                    # print('module trigger counter (starts at 0, when TRA or WRITE starts): ', trig_count)
                    # print('channel trigger delay relative to time stamp [µs]: ', trig_delay)
                    # print('absolute time [s] (computer time timeval.tv_sec): ', abs_time_s)
                    # print('absolute time [us] (computer time timeval.tv_us): ', abs_time_mus)
                    # print('Delay of channel trigger to testpulse [us]: ', delay_ch_tp)
                    # print('time stamp of module trigger low word (10 MHz clock, 0 @ START WRITE ): ', time_low)
                    # print('time stamp of module trigger high word (10 MHz clock, 0 @ START WRITE ): ', time_high)
                    # print('number of qdc events accumulated until digitizer trigger: ', qcd_events)
                    # print('measuring hours (0 @ START WRITE): ', hours)
                    # print('accumulated dead time of channel [s] (0 @ START WRITE): ', dead_time)
                    # print('test pulse amplitude (0. for pulses, (0.,10.] for test pulses, >10. for control pulses): ', test_pulse_amplitude)
                    # print('DAC output of control program (proportional to heater power): ', dac_output)

                    # print the dvm channels
                    for i in range(dvm_channels):
                        print('DVM channel {} : {}'.format(i, dvm[i]))

    # Converts a bck to a hdf5 for one module with 2 or 3 channels
    def convert_dataset(self, path_rdt,
                        fname, path_h5,
                        tpa_list=[0],
                        calc_mp=True, calc_fit=False,
                        calc_sev=False, processes=4):

        print('Start converting.')

        gen_dataset_from_rdt(path_rdt=path_rdt,
                             fname=fname,
                             path_h5=path_h5,
                             phonon_channel=self.channels[0],
                             light_channel=self.channels[1],
                             tpa_list=tpa_list,
                             calc_mp=calc_mp,
                             calc_fit=calc_fit,
                             calc_sev=calc_sev,
                             processes=processes
                             )

        print('Hdf5 dataset created in  {}'.format(path_h5))

        if self.nmbr_channels == 2:
            self.path_h5 = "{}{}-P_Ch{}-L_Ch{}.h5".format(path_h5, fname, self.channels[0],
                                                          self.channels[1])
            self.fname = fname

        else:
            raise NotImplementedError('Only for two channels implemented!')

        print('Filepath and -name saved.')

    def set_filepath(self, path_h5, fname):

        if self.nmbr_channels == 2:
            self.path_h5 = "{}/run{}_{}/{}-P_Ch{}-L_Ch{}.h5".format(path_h5, self.run, self.module,
                                                                    fname, self.channels[0],
                                                                    self.channels[1])
            self.fname = fname

        else:
            raise NotImplementedError('Only for two channels implemented!')

    # Recalculate MP
    def recalc_mp(self, type, path_h5=None, processes=4):

        if not path_h5:
            path_h5 = self.path_h5

        h5f = h5py.File(path_h5, 'r+')
        events = h5f[type]

        print('CALCULATE MAIN PARAMETERS.')

        with Pool(processes) as p:  # basically a for loop running on 4 processes
            p_mainpar_list_event = p.map(
                calc_main_parameters, events['event'][0, :, :])
            l_mainpar_list_event = p.map(
                calc_main_parameters, events['event'][1, :, :])
        mainpar_event = np.array([[o.getArray() for o in p_mainpar_list_event],
                                  [o.getArray() for o in l_mainpar_list_event]])

        events['mainpar'][...] = mainpar_event

    # Recalculate Fit
    def recalc_fit(self, path):
        raise NotImplementedError('Not implemented.')

    def recalc_sev(self,
                   use_labels=True,
                   pulse_height_intervall=[0.5, 1.5],
                   left_right_cutoff=None,
                   rise_time_intervall=None,
                   decay_time_intervall=None,
                   onset_intervall=None,
                   remove_offset=True,
                   verb=True):

        h5f = h5py.File(self.path_h5, 'r+')
        events = h5f['events']['event']
        mainpar = h5f['events']['mainpar']

        if use_labels:
            labels = h5f['events']['labels']
        else:
            labels = [None, None]

        # [pulse_height, t_zero, t_rise, t_max, t_decaystart, t_half, t_end, offset, linear_drift, quadratic_drift]
        p_stdevent_pulse, p_stdevent_fitpar = generate_standard_event(events=events[0, :, :],
                                                                      main_parameters=mainpar[0, :, :],
                                                                      labels=labels[0],
                                                                      pulse_height_intervall=pulse_height_intervall,
                                                                      left_right_cutoff=left_right_cutoff,
                                                                      rise_time_intervall=rise_time_intervall,
                                                                      decay_time_intervall=decay_time_intervall,
                                                                      onset_intervall=onset_intervall,
                                                                      remove_offset=remove_offset,
                                                                      verb=verb)

        l_stdevent_pulse, l_stdevent_fitpar = generate_standard_event(events=events[1, :, :],
                                                                      main_parameters=mainpar[1, :, :],
                                                                      labels=labels[1],
                                                                      pulse_height_intervall=pulse_height_intervall,
                                                                      left_right_cutoff=left_right_cutoff,
                                                                      rise_time_intervall=rise_time_intervall,
                                                                      decay_time_intervall=decay_time_intervall,
                                                                      onset_intervall=onset_intervall,
                                                                      remove_offset=remove_offset,
                                                                      verb=verb)

        stdevent = h5f.require_group('stdevent')

        stdevent.require_dataset('event',
                                 shape=(2, len(p_stdevent_pulse)),
                                 dtype='f')
        stdevent['event'][...] = np.array([p_stdevent_pulse, l_stdevent_pulse])
        stdevent.require_dataset('fitpar',
                                 shape=(2, len(p_stdevent_fitpar)),
                                 dtype='f')
        stdevent['fitpar'][...] = np.array([p_stdevent_fitpar, l_stdevent_fitpar])

        # description of the fitparameters (data=column_in_fitpar)
        stdevent['fitpar'].attrs.create(name='t_0', data=0)
        stdevent['fitpar'].attrs.create(name='A_n', data=1)
        stdevent['fitpar'].attrs.create(name='A_t', data=2)
        stdevent['fitpar'].attrs.create(name='tau_n', data=3)
        stdevent['fitpar'].attrs.create(name='tau_in', data=4)
        stdevent['fitpar'].attrs.create(name='tau_t', data=5)

        mp = np.array([calc_main_parameters(p_stdevent_pulse).getArray(),
                       calc_main_parameters(l_stdevent_pulse).getArray()])

        stdevent.require_dataset('mainpar',
                                 shape=mp.shape,
                                 dtype='f',
                                 data=mp)
        # description of the mainpar (data=col_in_mainpar)
        stdevent['mainpar'].attrs.create(name='pulse_height', data=0)
        stdevent['mainpar'].attrs.create(name='t_zero', data=1)
        stdevent['mainpar'].attrs.create(name='t_rise', data=2)
        stdevent['mainpar'].attrs.create(name='t_max', data=3)
        stdevent['mainpar'].attrs.create(name='t_decaystart', data=4)
        stdevent['mainpar'].attrs.create(name='t_half', data=5)
        stdevent['mainpar'].attrs.create(name='t_end', data=6)
        stdevent['mainpar'].attrs.create(name='offset', data=7)
        stdevent['mainpar'].attrs.create(name='linear_drift', data=8)
        stdevent['mainpar'].attrs.create(name='quadratic_drift', data=9)

        print('SEV updated.')

        h5f.close()

    def recalc_of(self):

        h5f = h5py.File(self.path_h5, 'r+')
        p_stdevent_pulse = h5f['stdevent']['event'][0]
        p_mean_nps = h5f['noise']['nps'][0]
        l_stdevent_pulse = h5f['stdevent']['event'][1]
        l_mean_nps = h5f['noise']['nps'][1]

        print('CREATE OPTIMUM FILTER.')

        of = np.array([optimal_transfer_function(p_stdevent_pulse, p_mean_nps),
                       optimal_transfer_function(l_stdevent_pulse, l_mean_nps)])

        optimumfilter = h5f.require_group('optimumfilter')
        optimumfilter.require_dataset('optimumfilter',
                                      shape=of.shape,
                                      dtype='f')

        optimumfilter['optimumfilter'][...] = of

        print('OF updated.')

        h5f.close()

    # Import label CSV file in hdf5 file
    def import_labels(self, path_labels, path_h5=None):

        if not path_h5:
            path_h5 = self.path_h5

        path_labels = '{}/run{}_{}/labels_{}_events.csv'.format(
            path_labels, self.run, self.module, self.fname)

        h5f = h5py.File(path_h5, 'r+')

        if path_labels != '' and os.path.isfile(path_labels):
            labels_event = np.genfromtxt(path_labels)
            labels_event = labels_event.astype('int32')
            length = len(labels_event)
            labels_event.resize((2, int(length / 2)))

            print(h5f.keys())

            events = h5f['events']

            if "labels" in events:
                events['labels'][...] = labels_event
                print('Edited Labels.')

            else:
                events.create_dataset('labels', data=labels_event)
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

    # Plot the SEV
    def show_SEV(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        # plot
        plt.close()
        plt.subplot(211)
        plt.plot(f['stdevent']['event'][0], color='blue')
        plt.title('Phonon SEV')
        plt.subplot(212)
        plt.plot(f['stdevent']['event'][1], color='red')
        plt.title('Light SEV')
        plt.show(block=block)

    # Plot the NPS
    def show_NPS(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        # plot
        plt.close()
        plt.subplot(211)
        plt.loglog(f['noise']['nps'][0], color='blue')
        plt.title('Phonon NPS')
        plt.subplot(212)
        plt.loglog(f['noise']['nps'][1], color='red')
        plt.title('Light NPS')
        plt.show(block=block)

    # Plot the OF
    def show_OF(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        of = f['optimumfilter']['optimumfilter']
        of = np.abs(of)**2

        # plot
        plt.close()
        plt.subplot(211)
        plt.loglog(of[0], color='blue')
        plt.title('Phonon OF')
        plt.subplot(212)
        plt.loglog(of[1], color='red')
        plt.title('Light OF')
        plt.show(block=block)

    # show histogram of main parameter
    def show_hist(self):
        # choose which mp to plot
        raise NotImplementedError('Not Implemented.')

    # show light yield plot
    def show_LY(self):
        # choose which labels to plot
        # choose which channels (e.g. for Gode modules)
        raise NotImplementedError('Not Implemented.')

    # calc stdevent testpulses
    def calc_SEV_tp(self):
        raise NotImplementedError('Not Implemented.')

    # calc stdevent carrier
    def calc_SEV_carrier(self):
        raise NotImplementedError('Not Implemented.')

    # Simulate Dataset with specific classes
    def simulate_fakenoise_dataset(self, classes_size):
        raise NotImplementedError('Not implemented.')

    # Simulate Dataset with real noise
    def simulate_realnoise_dataset(self, path_noise, classes_size):
        raise NotImplementedError('Not implemented.')
