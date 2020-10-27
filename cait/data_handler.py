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
from .fit._templates import pulse_template
from .fit._pm_fit import fit_pulse_shape
from .features._ts_feat import calc_ts_features
from functools import partial


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class DataHandler:

    def __init__(self, run, module, channels, record_length, sample_frequency=25000):
        # ask user things like which detector working on etc
        if len(channels) != 2:
            raise NotImplementedError('Only for 2 channels implemented.')
        self.run = run
        self.module = module
        self.record_length = record_length
        self.nmbr_channels = len(channels)
        self.channels = channels
        self.sample_frequency = sample_frequency

        if self.nmbr_channels == 2:
            self.channel_names = ['Phonon', 'Light']
            self.colors = ['red', 'blue']
        elif self.nmbr_channels == 3:
            self.channel_names = ['Channel 1', 'Channel 2', 'Channel 3']
            self.colors = ['red', 'red', 'blue']

        print('DataHandler Instance created.')

    # -----------------------------------------------------------
    # PROCESS RAW DATA; CREATE OR LOAD RAW DATA DATASETS
    # -----------------------------------------------------------

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

    def set_filepath(self, path_h5: str, fname: str):
        """
        Set the path to the bck_XXX.hdf5 file for further processing.

        :param path_h5: String to directory that contains the runXY_Module folders
        :param fname: String, usually something like bck_xxx
        :return: -
        """

        # check if the channel number matches the file, otherwise error
        if self.nmbr_channels == 2:
            self.path_h5 = "{}/run{}_{}/{}-P_Ch{}-L_Ch{}.h5".format(path_h5, self.run, self.module,
                                                                    fname, self.channels[0],
                                                                    self.channels[1])
            self.fname = fname

        else:
            raise NotImplementedError('Only for two channels implemented!')


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


    # -----------------------------------------------------------
    # FEATURE CALCULATION
    # -----------------------------------------------------------

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
    def recalc_fit(self, path_h5=None, type='events', processes=4):

        if type not in ['events', 'testpulses']:
            raise NameError('Type must be events or testpulses.')

        if not path_h5:
            path_h5 = self.path_h5

        h5f = h5py.File(path_h5, 'r+')
        events = h5f[type]['event'] # TODO is this correct?

        print('CALCULATE FIT.')

        # get start values from SEV fit if exists
        try:
            if type == 'events':
                sev_fitpar = h5f['stdevent']['fitpar']
                p_fit_pm = partial(fit_pulse_shape, x0=sev_fitpar[0])
                l_fit_pm = partial(fit_pulse_shape, x0=sev_fitpar[1])
            else:
                raise NameError('This is only to break the loop, bc type is not events.')
        except NameError:
            p_fit_pm = fit_pulse_shape
            l_fit_pm = fit_pulse_shape

        with Pool(processes) as p:
            p_fitpar_event = np.array(
                p.map(p_fit_pm, events[0, :, :]))
            l_fitpar_event = np.array(
                p.map(l_fit_pm, events[1, :, :]))

        fitpar_event = np.array([p_fitpar_event, l_fitpar_event])

        events.require_dataset('fitpar',
                               shape=fitpar_event.shape,
                               dtype='f')

        events['fitpar'][...] = fitpar_event

    def recalc_sev(self,
                   use_labels=True,
                   pulse_height_intervall=[0.5, 1.5],
                   left_right_cutoff=None,
                   rise_time_intervall=None,
                   decay_time_intervall=None,
                   onset_intervall=None,
                   remove_offset=True,
                   verb=True,
                   scale_fit_height=True,
                   sample_length=0.04):

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
                                                                      verb=verb,
                                                                      scale_fit_height=scale_fit_height,
                                                                      sample_length=sample_length)

        l_stdevent_pulse, l_stdevent_fitpar = generate_standard_event(events=events[1, :, :],
                                                                      main_parameters=mainpar[1, :, :],
                                                                      labels=labels[1],
                                                                      pulse_height_intervall=pulse_height_intervall,
                                                                      left_right_cutoff=left_right_cutoff,
                                                                      rise_time_intervall=rise_time_intervall,
                                                                      decay_time_intervall=decay_time_intervall,
                                                                      onset_intervall=onset_intervall,
                                                                      remove_offset=remove_offset,
                                                                      verb=verb,
                                                                      scale_fit_height=scale_fit_height,
                                                                      sample_length=sample_length)

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

    # calculate TS Features
    def calc_features(self, type='events', downsample=None):

        f = h5py.File(self.path_h5, 'r+')
        events = np.array(f[type]['event'])
        features = []

        if downsample is None:
            downsample = self.down

        for c in range(self.nmbr_channels):
            features.append(calc_ts_features(events=events[c],
                             nmbr_channels=self.nmbr_channels,
                             nmbr_events=len(events[0]),
                             record_length=self.record_length,
                             down=downsample,
                             sample_frequency=self.sample_frequency,
                             scaler=None))

        features = np.array(features)

        print('Features calculated.')

        f[type].require_dataset('ts_features',
                               shape=features.shape,
                               dtype='f')

        f[type]['ts_features'][...] = features

    # -----------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------

    # Plot the SEV
    def show_SEV(self, type='stdevent', block=True, sample_length=0.04):
        f = h5py.File(self.path_h5, 'r')
        sev = f[type]['event']
        sev_fitpar = f[type]['fitpar']

        t = (np.arange(0, self.record_length, dtype=float) - self.record_length / 4) * sample_length

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.plot(t, sev[i], color=self.colors[i])
            plt.plot(t, pulse_template(t, *sev_fitpar[i]), color='orange')
            plt.title(ch + ' ' + type)

        plt.show(block=block)

    # Plot the NPS
    def show_NPS(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.loglog(f['noise']['nps'][i], color=self.colors[i])
            plt.title(ch + ' NPS')

        plt.show(block=block)

    # Plot the OF
    def show_OF(self, block=True):
        f = h5py.File(self.path_h5, 'r')

        of = f['optimumfilter']['optimumfilter']
        of = np.abs(of) ** 2

        # plot
        plt.close()

        for i, ch in enumerate(self.channel_names):
            plt.subplot(2, 1, i + 1)
            plt.loglog(of[i], color=self.colors[i])
            plt.title(ch + ' OF')

        plt.show(block=block)

    # show histogram of main parameter
    def show_hist(self,
                  which_mp='pulse_height',
                  which_channel=0,
                  type='events',
                  which_labels=None,
                  bins=100,
                  block=False):
        # pulse_height
        # t_zero
        # t_rise
        # t_max
        # t_decaystart
        # t_half
        # t_end
        # offset
        # linear_drift
        # quadratic_drift

        f_h5 = h5py.File(self.path_h5, 'r')
        nmbr_mp = f_h5[type]['mainpar'].attrs[which_mp]
        par = f_h5[type]['mainpar'][which_channel, :, nmbr_mp]
        if which_labels is not None:
            pars = []
            for lab in which_labels:
                pars.append(par[f_h5[type]['labels'][which_channel] == lab])

        # choose which mp to plot
        plt.close()
        if which_labels is not None:
            for p, l in zip(pars, which_labels):
                plt.hist(p, bins=bins, label='Label ' + str(l), alpha=0.8)
        else:
            plt.hist(par, bins=bins, color=self.colors[which_channel])
        plt.title(which_mp + ' Channel ' + str(which_channel))
        plt.show(block=block)

    # show light yield plot
    def show_LY(self):
        # choose which labels to plot
        # choose which channels (e.g. for Gode modules)
        raise NotImplementedError('Not Implemented.')

    # -----------------------------------------------------------
    # CALCULATE FEATURES FOR SIMULATION
    # -----------------------------------------------------------

    # calc stdevent testpulses
    def calc_SEV_tp(self,
                    pulse_height_intervall=[0.5, 1.5],
                    left_right_cutoff=None,
                    rise_time_intervall=None,
                    decay_time_intervall=None,
                    onset_intervall=None,
                    remove_offset=True,
                    verb=True,
                    scale_fit_height=True,
                    sample_length=0.04):

        h5f = h5py.File(self.path_h5, 'r+')
        events = h5f['testpulses']['event']
        mainpar = h5f['testpulses']['mainpar']

        sev = []

        # fix the issue with different arguments for different channels
        inp = [left_right_cutoff, rise_time_intervall, decay_time_intervall, onset_intervall]
        for i, var in enumerate(inp):
            if var is None:
                inp[i] = [None for i in range(self.nmbr_channels)]

        for c in range(self.nmbr_channels):
            sev.append(generate_standard_event(events=events[c, :, :],
                                               main_parameters=mainpar[c, :, :],
                                               pulse_height_intervall=pulse_height_intervall[c],
                                               left_right_cutoff=inp[0][c],
                                               rise_time_intervall=inp[1][c],
                                               decay_time_intervall=inp[2][c],
                                               onset_intervall=inp[3][c],
                                               remove_offset=remove_offset,
                                               verb=verb,
                                               scale_fit_height=scale_fit_height,
                                               sample_length=sample_length))

        sev_tp = h5f.require_group('stdevent_tp')

        sev_tp.require_dataset('event',
                               shape=(self.nmbr_channels, len(sev[0][0])),  # this is then length of sev
                               dtype='f')
        sev_tp['event'][...] = np.array([x[0] for x in sev])
        sev_tp.require_dataset('fitpar',
                               shape=(self.nmbr_channels, len(sev[0][1])),
                               dtype='f')
        sev_tp['fitpar'][...] = np.array([x[1] for x in sev])

        # description of the fitparameters (data=column_in_fitpar)
        sev_tp['fitpar'].attrs.create(name='t_0', data=0)
        sev_tp['fitpar'].attrs.create(name='A_n', data=1)
        sev_tp['fitpar'].attrs.create(name='A_t', data=2)
        sev_tp['fitpar'].attrs.create(name='tau_n', data=3)
        sev_tp['fitpar'].attrs.create(name='tau_in', data=4)
        sev_tp['fitpar'].attrs.create(name='tau_t', data=5)

        mp = np.array([calc_main_parameters(x[0]).getArray() for x in sev])

        sev_tp.require_dataset('mainpar',
                               shape=mp.shape,
                               dtype='f')

        sev_tp['mainpar'][...] = mp

        # description of the mainpar (data=col_in_mainpar)
        sev_tp['mainpar'].attrs.create(name='pulse_height', data=0)
        sev_tp['mainpar'].attrs.create(name='t_zero', data=1)
        sev_tp['mainpar'].attrs.create(name='t_rise', data=2)
        sev_tp['mainpar'].attrs.create(name='t_max', data=3)
        sev_tp['mainpar'].attrs.create(name='t_decaystart', data=4)
        sev_tp['mainpar'].attrs.create(name='t_half', data=5)
        sev_tp['mainpar'].attrs.create(name='t_end', data=6)
        sev_tp['mainpar'].attrs.create(name='offset', data=7)
        sev_tp['mainpar'].attrs.create(name='linear_drift', data=8)
        sev_tp['mainpar'].attrs.create(name='quadratic_drift', data=9)

        print('TP SEV calculated.')

        h5f.close()

    # calc stdevent carrier
    def calc_SEV_carrier(self):
        raise NotImplementedError('Not Implemented.')

    def calc_bl_coefficients(self):
        raise NotImplementedError('Not Implemented.')

    # -----------------------------------------------------------
    # SIMULATE DATA
    # -----------------------------------------------------------

    # Simulate Dataset with specific classes
    def simulate_fakenoise_pulses(self,
                                  path,
                                  fname,
                                  size,
                                  ph_interval=[0, 1],
                                  discrete_ph=None,
                                  chunk_size=1000):

        # create file handle
        filename = '{}/{}_{}/{}'.format(path, self.run, self.module, fname)
        f = h5py.File( filename , 'w')

        nmbr_chunks = int(size / chunk_size)
        residual = size - chunk_size * nmbr_chunks

        if residual == 0:
            range_up = nmbr_chunks
        else:
            range_up = nmbr_chunks + 1

        for counter in range(range_up):
            print('###################################################')
            print('COUNTER: ', counter)
            if counter < nmbr_chunks:
                current_size = chunk_size
            else:
                current_size = residual

            # -------------------------------------------------
            # GET BASELINES
            # -------------------------------------------------

            if counter == 0:
                ...

        # simulate baselines

        # add pulses

        raise NotImplementedError('Not implemented.')

    # Simulate Dataset with real noise
    def simulate_realnoise_dataset(self, path_noise, classes_size):
        raise NotImplementedError('Not implemented.')
