# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import struct
from ..data._gen_h5 import gen_dataset_from_rdt
import h5py
from ..data._raw import read_rdt_file

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class RdtMixin(object):
    """
    Mixin for the DataHandler, includes methods to process the RDT files
    """

    # Checkout RDT File if Channel exists etc
    def checkout_rdt(self, path_rdt, read_events=100, tpa_list=[0, 1, -1], dvm_channels=0, verb=False,
                     ints_in_header=7):
        """
        Prints the channel numbers of a number of events in the rdt File, from the beginning

        :param path_rdt: string, full path to the rdt file e.g. "data/bcks/bck_001.rdt"
        :param read_events: int, number of events to read from file, if -1 read all events
        :param tpa_list: list, the tpas that are to read from the file, 0 -> events, -1 -> noise, 1 -> tp
        :param dvm_channels: int, number of dvm channels
        :return: -
        """
        # initialize dvm and event arrays
        dvm = np.zeros(dvm_channels, dtype=float)
        event = np.zeros(self.record_length, dtype=np.short)

        with open(path_rdt, "rb") as f:
            for nmbr_event in range(read_events):

                dummies = []

                try:
                    # read all header infos of the event
                    detector_nmbr = struct.unpack('i', f.read(4))[0]
                    coincide_pulses = struct.unpack('i', f.read(4))[0]
                    trig_count = struct.unpack('i', f.read(4))[0]
                    trig_delay = struct.unpack('i', f.read(4))[0]
                    abs_time_s = struct.unpack('i', f.read(4))[0]
                    abs_time_mus = struct.unpack('i', f.read(4))[0]
                    if ints_in_header == 7:
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

                except struct.error as err:
                    print(err)
                    print('File finished, returning to main.')
                    return

                # print all headerinfos

                if (test_pulse_amplitude in tpa_list) or (test_pulse_amplitude > 0 and 1 in tpa_list):
                    print(
                        '#############################################################')
                    print('EVENT NUMBER: ', nmbr_event)

                    print('detector number (starting at 0): ', detector_nmbr)

                    if verb:
                        print('number of coincident pulses in digitizer module: ', coincide_pulses)
                        print('module trigger counter (starts at 0, when TRA or WRITE starts): ', trig_count)
                        print('channel trigger delay relative to time stamp [µs]: ', trig_delay)
                        print('absolute time [s] (computer time timeval.tv_sec): ', abs_time_s)
                        print('absolute time [us] (computer time timeval.tv_us): ', abs_time_mus)
                        if ints_in_header == 7:
                            print('Delay of channel trigger to testpulse [us]: ', delay_ch_tp)
                        print('time stamp of module trigger low word (10 MHz clock, 0 @ START WRITE ): ', time_low)
                        print('time stamp of module trigger high word (10 MHz clock, 0 @ START WRITE ): ', time_high)
                        print('number of qdc events accumulated until digitizer trigger: ', qcd_events)
                    print('measuring hours (0 @ START WRITE): ', hours)
                    if verb:
                        print('accumulated dead time of channel [s] (0 @ START WRITE): ', dead_time)
                        print('test pulse amplitude (0. for pulses, (0.,10.] for test pulses, >10. for control pulses): ', test_pulse_amplitude)
                        print('DAC output of control program (proportional to heater power): ', dac_output)

                    # print the dvm channels
                    for i in range(dvm_channels):
                        print('DVM channel {} : {}'.format(i, dvm[i]))

    # checkout con file
    def checkout_con(self, path_con, read_events=100, ints_in_header=7):
        """
        TODO

        :param path_con:
        :type path_con:
        :param read_events:
        :type read_events:
        :param ints_in_header:
        :type ints_in_header:
        :return:
        :rtype:
        """
        record = np.dtype([('detector_nmbr', 'int32'),
                           ('pulse_height', 'float32'),
                           ('time_stamp_low', 'uint32'),
                           ('time_stamp_high', 'uint32'),
                           ('dead_time', 'float32'),
                           ('mus_since_last_tp', 'int32', (int(ints_in_header == 7),)),
                           ])

        cons = np.fromfile(path_con, dtype=record, offset=12)

        if read_events > len(cons):
            read_events = len(cons)

        print('{} control pulses read from CON file.'.format(read_events))

        if ints_in_header == 7:
            print(" \tdetector_nmbr,\t \tpulse_height, \ttime_stamp_low, \ttime_stamp_high, \tdead_time, \tmus_since_last_tp")
            for i in range(read_events):
                print('{}\t{}\t\t{:.3}\t\t{}\t\t{}\t\t\t{:.3}\t{}'.format(i+1, cons['detector_nmbr'][i],
                                                      cons['pulse_height'][i],
                                                      cons['time_stamp_low'][i],
                                                      cons['time_stamp_high'][i],
                                                      cons['dead_time'][i],
                                                      cons['mus_since_last_tp'][i],
                                                      ))
        else:
            print("\tdetector_nmbr, \tpulse_height, \ttime_stamp_low, \ttime_stamp_high, \tdead_time")
            for i in range(read_events):
                print('{}\t{}\t\t{:.3}\t\t{}\t\t{}\t\t\t{:.3}'.format(i+1, cons['detector_nmbr'][i],
                                                      cons['pulse_height'][i],
                                                      cons['time_stamp_low'][i],
                                                      cons['time_stamp_high'][i],
                                                      cons['dead_time'][i],
                                                      ))

    # checkout dig file
    def checkout_dig(self, path_dig, read_events=100):
        """
        TODO

        :param path_dig:
        :type path_dig:
        :param read_events:
        :type read_events:
        :return:
        :rtype:
        """
        dig = np.dtype([
            ('stamp', np.uint64),
            ('bank', np.uint32),
            ('bank2', np.uint32),
        ])

        stamps = np.fromfile(path_dig, dtype=dig)

        if read_events > len(stamps):
            read_events = len(stamps)

        print("stamp, bank, _")
        for i in range(read_events):
            print(stamps[i])

    # checkout test stamps file
    def checkout_test(self, path_test, read_events=100):
        """
        TDOO

        :param path_test:
        :type path_test:
        :param read_events:
        :type read_events:
        :return:
        :rtype:
        """

        teststamp = np.dtype([
            ('stamp', np.uint64),
            ('tpa', np.float32),
            ('tpch', np.uint32),
        ])

        stamps = np.fromfile(path_test, dtype=teststamp)

        if read_events > len(stamps):
            read_events = len(stamps)

        print("stamp, tpa, tpch")
        for i in range(read_events):
            print(stamps[i])

    def checkout_mon(self, path_mon, read_events=5):
        """
        TODO

        :param path_mon:
        :type path_mon:
        :param read_events:
        :type read_events:
        :return:
        :rtype:
        """

        header = np.dtype([('file_version', '|S12'),
                           ('time_s', 'int64'),
                           ('time_mus', 'int64'),
                           ('nmbr_par', 'int32'),
                           ])

        mon_head = np.fromfile(path_mon, dtype=header, count=1)
        print(mon_head)

        mon_par_names = np.fromfile(path_mon, dtype='|S48', count=mon_head[0]['nmbr_par'], offset=mon_head.nbytes)

        record = np.dtype([('time_s', 'int64'),
                           ('time_mus', 'int64'),
                           ] +
                          [(n.decode(), 'float32') for n in mon_par_names])

        mon_pars = np.fromfile(path_mon, dtype=record, count=read_events, offset=mon_head.nbytes + mon_par_names.nbytes)

        print('Nmr records: ', len(mon_pars))
        print('Names: ', record.names)
        print('Read Pars: ', mon_pars)


    # Converts a bck to a hdf5 for one module with 2 or 3 channels
    def convert_dataset(self, path_rdt,
                        fname, path_h5,
                        tpa_list=[0, 1, -1],
                        calc_mp=True, calc_fit=False,
                        calc_sev=False, calc_nps=True,
                        processes=4,
                        event_dtype='float32',
                        ints_in_header=7,
                        lazy_loading=True,
                        ):
        """
        Wrapper for the gen_dataset_from_rdt function, creates HDF5 dataset from Rdt file

        :param path_rdt: string, path to the rdt file e.g. "data/bcks/"
        :param fname: string, name of the file e.g. "bck_001"
        :param path_h5: string, path where the h5 file is saved e.g. "data/hdf5s%"
        :param tpa_list: list, the test pulse amplitudes to save
        :param calc_mp: bool, if True the main parameters for all events are calculated and stored
        :param calc_fit: bool, if True the parametric fit for all events is calculated and stored
        :param calc_sev: bool, if True the standard event for all event channels is calculated
        :param processes: int, the number of processes that is used for the code execution
        :param chunk_size: int, the init size of the arrays, should ideally
            be a bit more than we read events
        :param event_dtype: string, datatype to save the events with
        :return: -
        """

        print('Start converting.')

        gen_dataset_from_rdt(path_rdt=path_rdt,
                             fname=fname,
                             path_h5=path_h5,
                             channels=self.channels,
                             tpa_list=tpa_list,
                             calc_mp=calc_mp,
                             calc_fit=calc_fit,
                             calc_sev=calc_sev,
                             calc_nps=calc_nps,
                             processes=processes,
                             event_dtype=event_dtype,
                             ints_in_header=ints_in_header,
                             sample_frequency=self.sample_frequency,
                             lazy_loading=lazy_loading,
                             )

        print('Hdf5 dataset created in  {}'.format(path_h5))

        self.path_directory = path_h5

        if self.nmbr_channels == 2:
            self.path_h5 = "{}{}-P_Ch{}-L_Ch{}.h5".format(path_h5, fname, self.channels[0],
                                                          self.channels[1])

        else:
            path = "{}{}".format(path_h5, fname)
            for i, c in enumerate(self.channels):
                path += '-{}_Ch{}'.format(i + 1, c)
            path += ".h5"
            self.path_h5 = path
        self.fname = fname


        print('Filepath and -name saved.')

    # include rdt infos to existing hdf5
    def include_rdt(self, path_data, fname, channels, ints_in_header=7, tpa_list=[0, 1, -1],
                    event_dtype='float32', lazy_loading=True):
        """
        TODO

        :param path_data:
        :type path_data:
        :param fname:
        :type fname:
        :param channels:
        :type channels:
        :param ints_in_header:
        :type ints_in_header:
        :param tpa_list:
        :type tpa_list:
        :param event_dtype:
        :type event_dtype:
        :return:
        :rtype:
        """
        print('Accessing RDT File ...')

        metainfo, pulse = \
            read_rdt_file(fname=fname,
                          path=path_data,
                          channels=channels,
                          store_as_int=False,
                          ints_in_header=ints_in_header,
                          lazy_loading=lazy_loading
                          )

        with h5py.File(self.path_h5, 'r+') as h5f:
            # events
            if 0.0 in tpa_list:

                metainfo_event = metainfo[:, metainfo[0, :, 12] == 0, :]
                pulse_event = pulse[:, metainfo[0, :, 12] == 0, :]

                nmbr_events = len(metainfo_event[0])
                print('Adding {} triggered Events.'.format(nmbr_events))

                events = h5f.require_group('events')
                events.require_dataset('event', shape=pulse_event.shape, dtype=event_dtype)
                events.require_dataset('hours', shape=metainfo_event[0, :, 10].shape, dtype=float)
                events.require_dataset('time_s', shape=metainfo_event[0, :, 4].shape, dtype='int32')
                events.require_dataset('time_mus', shape=metainfo_event[0, :, 5].shape, dtype='int32')

                events['event'][...] = np.array(pulse_event, dtype=event_dtype)
                events['hours'][...] = np.array(metainfo_event[0, :, 10])
                events['time_s'][...] = np.array(metainfo_event[0, :, 4], dtype='int32')
                events['time_mus'][...] = np.array(metainfo_event[0, :, 5], dtype='int32')

            # noise
            if -1.0 in tpa_list:

                metainfo_noise = metainfo[:, metainfo[0, :, 12] == -1.0, :]
                pulse_noise = pulse[:, metainfo[0, :, 12] == -1.0, :]

                nmbr_noise = len(metainfo_noise[0])
                print('Adding {} Noise Events.'.format(nmbr_noise))

                noise = h5f.require_group('noise')
                noise.require_dataset('event', shape=pulse_noise.shape, dtype=event_dtype)
                noise.require_dataset('hours', shape=metainfo_noise[0, :, 10].shape, dtype=float)
                noise.require_dataset('time_s', shape=metainfo_noise[0, :, 4].shape, dtype='int32')
                noise.require_dataset('time_mus', shape=metainfo_noise[0, :, 5].shape, dtype='int32')

                noise['event'][...] = np.array(pulse_noise, dtype=event_dtype)
                noise['hours'][...] = np.array(metainfo_noise[0, :, 10])
                noise['time_s'][...] = np.array(metainfo_noise[0, :, 4], dtype='int32')
                noise['time_mus'][...] = np.array(metainfo_noise[0, :, 5], dtype='int32')

            # testpulses
            if any(el > 0 for el in tpa_list):

                tp_list = np.logical_and(
                    metainfo[0, :, 12] != -1.0, metainfo[0, :, 12] != 0.0)

                metainfo_tp = metainfo[:, tp_list, :]
                pulse_tp = pulse[:, tp_list, :]

                nmbr_tp = len(metainfo_tp[0])
                print('Adding {} Testpulse Events.'.format(nmbr_tp))

                testpulses = h5f.require_group('testpulses')
                testpulses.require_dataset('event', shape=pulse_tp.shape, dtype=event_dtype)
                testpulses.require_dataset('hours', shape=metainfo_tp[0, :, 10].shape, dtype=float)
                testpulses.require_dataset('time_s', shape=metainfo_tp[0, :, 4].shape, dtype='int32')
                testpulses.require_dataset('time_mus', shape=metainfo_tp[0, :, 5].shape, dtype='int32')

                testpulses['event'][...] = np.array(pulse_tp, dtype=event_dtype)
                testpulses['hours'][...] = np.array(metainfo_tp[0, :, 10])
                testpulses['time_s'][...] = np.array(metainfo_tp[0, :, 4], dtype='int32')
                testpulses['time_mus'][...] = np.array(metainfo_tp[0, :, 5], dtype='int32')

        print('Done.')

    def include_con_file(self, path_con_file, ints_in_header=7):
        """
        TODO

        :param path_con_file:
        :type path_con_file:
        :param ints_in_header:
        :type ints_in_header:
        :return:
        :rtype:
        """

        print('Accessing CON File...')

        if ints_in_header == 7:
            clock_frequency = 1e7
        else:
            clock_frequency = 1.7e9

        # define data type for file read
        record = np.dtype([('detector_nmbr', 'int32'),
                           ('pulse_height', 'float32'),
                           ('time_stamp_low', 'uint32'),
                           ('time_stamp_high', 'uint32'),
                           ('dead_time', 'float32'),
                           ('mus_since_last_tp', 'int32', (int(ints_in_header == 7),)),
                           ])

        # get data from con file
        data = np.fromfile(path_con_file, dtype=record, offset=12)

        cond = data['detector_nmbr'] == self.channels[0]
        nmbr_cp = np.sum(cond)
        print('{} Control Pulses for channel {} in file.'.format(nmbr_cp, self.channels[0]))
        hours = (data['time_stamp_high'][cond] * 2 ** 32 + data['time_stamp_low'][cond]) / clock_frequency / 3600

        # create file handles
        with h5py.File(self.path_h5, 'r+') as f:
            f.require_group('controlpulses')
            cp_hours = f['controlpulses'].require_dataset(name='hours',
                                                          shape=(nmbr_cp,),
                                                          dtype=float)
            cp_hours[...] = hours
            cp_ph = f['controlpulses'].require_dataset(name='pulse_height',
                                                       shape=(self.nmbr_channels, nmbr_cp),
                                                       dtype=float)

            for i, c in enumerate(self.channels):
                cond = data['detector_nmbr'] == c

                # write data to file
                cp_ph[i, ...] = data['pulse_height'][cond]

        print('CON File included.')

    def include_mon(self, path_mon):
        """
        TODO

        :param path_mon:
        :type path_mon:
        :param read_events:
        :type read_events:
        :return:
        :rtype:
        """

        header = np.dtype([('file_version', '|S12'),
                           ('time_s', 'int64'),
                           ('time_mus', 'int64'),
                           ('nmbr_par', 'int32'),
                           ])

        mon_head = np.fromfile(path_mon, dtype=header, count=1)

        mon_par_names = np.fromfile(path_mon, dtype='|S48', count=mon_head[0]['nmbr_par'], offset=mon_head.nbytes)

        record = np.dtype([('time_s', 'int64'),
                           ('time_mus', 'int64'),
                           ] +
                          [(n.decode().replace('/', ' per '), 'float32') for n in mon_par_names])

        mon_pars = np.fromfile(path_mon, dtype=record, count=-1, offset=mon_head.nbytes + mon_par_names.nbytes)

        # create file handles
        with h5py.File(self.path_h5, 'r+') as f:
            mon = f.require_group('monitor')
            for name in record.names:
                if name in mon:
                    del mon[name]
                mon.create_dataset(name=name,
                                 data=mon_pars[name])

        print('MON File Included.')