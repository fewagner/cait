# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import struct
from ..data._gen_h5 import gen_dataset_from_rdt
from ..data._gen_h5_memsafe import gen_dataset_from_rdt_memsafe
import h5py
from ..data._raw import read_rdt_file, get_metainfo
import warnings
import time
import tracemalloc


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

        :param path_rdt: Full path to the rdt file e.g. "data/bcks/bck_001.rdt".
        :type path_rdt: string
        :param read_events: Number of events to read from file, if -1 read all events.
        :type read_events: int
        :param tpa_list: The tpas that are to read from the file, 0 -> events, -1 -> noise, 1 -> tp.
        :type tpa_list: list
        :param dvm_channels: Number of dvm channels.
        :type dvm_channels: int
        :param verb: If activated, all read parameters are printed.
        :type verb: bool
        :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
        :type ints_in_header: int
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
                    qdc_events = struct.unpack('I', f.read(4))[0]  # 'L'
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
                        print('number of qdc events accumulated until digitizer trigger: ', qdc_events)
                    print('measuring hours (0 @ START WRITE): ', hours)
                    if verb:
                        print('accumulated dead time of channel [s] (0 @ START WRITE): ', dead_time)
                        print(
                            'test pulse amplitude (0. for pulses, (0.,10.] for test pulses, >10. for control pulses): ',
                            test_pulse_amplitude)
                        print('DAC output of control program (proportional to heater power): ', dac_output)

                    # print the dvm channels
                    for i in range(dvm_channels):
                        print('DVM channel {} : {}'.format(i, dvm[i]))

    # checkout con file
    def checkout_con(self, path_con, read_events=100, ints_in_header=7):
        """
        Print the content of a *.con file.

        :param path_con: Path to the con file e.g. "data/bcks/".
        :type path_con: string
        :param read_events: The number of events to print from the file.
        :type read_events: int
        :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
        :type ints_in_header: int
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
            print(
                " \tdetector_nmbr,\t \tpulse_height, \ttime_stamp_low, \ttime_stamp_high, \tdead_time, \tmus_since_last_tp")
            for i in range(read_events):
                print('{}\t{}\t\t{:.3}\t\t{}\t\t{}\t\t\t{:.3}\t{}'.format(i + 1, cons['detector_nmbr'][i],
                                                                          cons['pulse_height'][i],
                                                                          cons['time_stamp_low'][i],
                                                                          cons['time_stamp_high'][i],
                                                                          cons['dead_time'][i],
                                                                          cons['mus_since_last_tp'][i],
                                                                          ))
        else:
            print("\tdetector_nmbr, \tpulse_height, \ttime_stamp_low, \ttime_stamp_high, \tdead_time")
            for i in range(read_events):
                print('{}\t{}\t\t{:.3}\t\t{}\t\t{}\t\t\t{:.3}'.format(i + 1, cons['detector_nmbr'][i],
                                                                      cons['pulse_height'][i],
                                                                      cons['time_stamp_low'][i],
                                                                      cons['time_stamp_high'][i],
                                                                      cons['dead_time'][i],
                                                                      ))

    # checkout dig file
    def checkout_dig(self, path_dig, read_events=100):
        """
        Print the content of a *.dig_stamps file.

        :param path_dig: Path to the dig file e.g. "data/bcks/*.dig_stamps".
        :type path_dig: string
        :param read_events: Number of events to read from file, if -1 read all events.
        :type read_events: int
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
        Print the content of a *.test_stamps file.

        :param path_test: Full path to the test file e.g. "data/bcks/*.test_stamps".
        :type path_test: string
        :param read_events: Number of events to read from file, if -1 read all events.
        :type read_events: int
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
        Print the content of a *.mon file.

        :param path_mon: Path to the mon file e.g. "data/bcks/*.mon".
        :type path_mon: string
        :param read_events: Number of events to read from file, if -1 read all events.
        :type read_events: int
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
    def convert_dataset(self,
                        path_rdt,
                        fname,
                        path_h5,
                        tpa_list=[0, 1, -1],
                        calc_mp=False,
                        calc_fit=False,
                        calc_sev=False,
                        calc_nps=False,
                        processes=4,
                        event_dtype='float32',
                        ints_in_header=7,
                        lazy_loading=True,
                        memsafe=True,
                        dvm_channels=0,
                        batch_size=1000,
                        trace=False,
                        indiv_tpas=False,
                        ):
        """
        Wrapper for the gen_dataset_from_rdt function, creates HDF5 dataset from Rdt file.

        :param path_rdt: Path to the rdt file e.g. "data/bcks/".
        :type path_rdt: string
        :param fname: Name of the file e.g. "bck_001".
        :type fname: string
        :param path_h5: Path where the h5 file is saved e.g. "data/hdf5s%".
        :type path_h5: string
        :param tpa_list: The test pulse amplitudes to save, if 1 is in the list, all positive values are included.
        :type tpa_list: list
        :param calc_mp: If True the main parameters for all events are calculated and stored.
        :type calc_mp: bool
        :param calc_fit: Not recommended! If True the parametric fit for all events is calculated and stored.
        :type calc_fit: bool
        :param calc_sev: Not recommended! If True the standard event for all event channels is calculated.
        :type calc_sev: bool
        :param calc_nps: If True the main parameters for all events are calculated and stored.
        :type calc_nps: bool
        :param processes: The number of processes that is used for the code execution.
        :type processes: int
        :param event_dtype: Datatype to save the events with.
        :type event_dtype: string
        :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
        :type ints_in_header: int
        :param lazy_loading: Recommended! If true, the data is loaded with memory mapping to avoid memory overflows.
        :type lazy_loading: bool
        :param dvm_channels: The number of DVM channels, this can be read in the PAR file.
        :type dvm_channels: int
        :param batch_size: The batch size for loading the samples from disk.
        :type batch_size: int
        :param memsafe: Recommended! This activates the version of data set conversion, which does not load all events
            into memory.
        :type memsafe: bool
        :param trace: Trace the runtime and memory consumption
        :type trace: bool
        :param individual_tpas: Write individual TPAs for the all channels. This results in a testpulseamplitude dataset
            of shape (nmbr_channels, nmbr_testpulses). Otherwise we have (nmbr_testpulses).
        :type individual_tpas: bool
        """

        assert self.channels is not None, 'To use this function, you need to specify the channel numbers either in the ' \
                                          'instanciation or when setting the file path for this instance!'

        print('Start converting.')

        if not memsafe:
            warnings.warn('Consider using the memsafe option! From the next release, it will be activated by default.')

            if trace:
                tracemalloc.start()
                start_time = time.time()

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

            if trace:
                print("--- %s seconds ---" % (time.time() - start_time))
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
                tracemalloc.stop()
        else:
            if calc_mp:
                warnings.warn('Memsafe implementation does not privide on-the-fly main paramter calculation. '
                              'Please call the calc_mp method instead.')
            if calc_fit:
                warnings.warn('Memsafe implementation does not privide on-the-fly parametric fit calculation. '
                              'Please call the calc_parametric_fit method instead.')
            if calc_sev:
                warnings.warn('Memsafe implementation does not privide on-the-fly sev calculation. '
                              'Please call the calc_sev method instead.')
            if calc_nps:
                warnings.warn('Memsafe implementation does not privide on-the-fly noise power spectrum calculation. '
                              'Please call the calc_nps method instead.')

            gen_dataset_from_rdt_memsafe(path_rdt=path_rdt,
                                         fname=fname,
                                         path_h5=path_h5,
                                         channels=self.channels,
                                         tpa_list=tpa_list,
                                         event_dtype=event_dtype,
                                         ints_in_header=ints_in_header,
                                         dvm_channels=dvm_channels,
                                         record_length=self.record_length,
                                         batch_size=batch_size,
                                         trace=trace,
                                         indiv_tpas=indiv_tpas,
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
    def include_rdt(self, path_data, fname, ints_in_header=7, tpa_list=[0, 1, -1],
                    event_dtype='float32', lazy_loading=True, origin=None, channels=None):
        """
        Read the content of an rdt file an add to HDF5.

        :param path_data: Path to the rdt file e.g. "data/bcks/*.rdt".
        :type path_data: string
        :param fname: Name of the file e.g. "bck_001".
        :type fname: string
        :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
        :type ints_in_header: int
        :param tpa_list: The test pulse amplitudes to save, if 1 is in the list, all positive values are included.
        :type tpa_list: list
        :param event_dtype: Datatype to save the events with.
        :type event_dtype: string
        :param lazy_loading: Recommended! If true, the data is loaded with memory mapping to avoid memory overflows.
        :type lazy_loading: bool
        :param origin: This is needed in case you add to merged dataset. In the merge, you can assign an origin data set
            with individual strings for all original files. If you provide an origin string here, the events are written
            at the corresponding position in the event array.
        :type origin: string
        """
        print('Accessing RDT File ...')

        assert self.channels is not None, 'To use this function, you need to specify the channel numbers either in the ' \
                                          'instanciation or when setting the file path for this instance!'

        if channels is None:
            channels = self.channels

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

                if origin is not None:
                    try:
                        idx = np.array([st.decode() == origin for st in events['origin'][:]]).nonzero()[0]
                    except:
                        idx = np.array([st == origin for st in events['origin'][:]]).nonzero()[0]
                    events.require_dataset('event',
                                           shape=(self.nmbr_channels, len(events['hours']), self.record_length),
                                           dtype=event_dtype)
                    events['event'][:, idx, :] = np.array(pulse_event, dtype=event_dtype)

                else:
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

                if origin is not None:
                    try:
                        idx = np.array([st.decode() == origin for st in noise['origin'][:]]).nonzero()[0]
                    except:
                        idx = np.array([st == origin for st in noise['origin'][:]]).nonzero()[0]
                    noise.require_dataset('event',
                                          shape=(self.nmbr_channels, len(noise['hours']), self.record_length),
                                          dtype=event_dtype)
                    noise['event'][:, idx, :] = np.array(pulse_noise, dtype=event_dtype)

                else:
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

                tp_list = metainfo[0, :, 12] > 0.0

                # np.logical_and(
                #     metainfo[0, :, 12] != -1.0, metainfo[0, :, 12] != 0.0)

                metainfo_tp = metainfo[:, tp_list, :]
                pulse_tp = pulse[:, tp_list, :]

                nmbr_tp = len(metainfo_tp[0])
                print('Adding {} Testpulse Events.'.format(nmbr_tp))

                testpulses = h5f.require_group('testpulses')

                if origin is not None:
                    try:
                        idx = np.array([st.decode() == origin for st in testpulses['origin'][:]]).nonzero()[0]
                    except:
                        idx = np.array([st == origin for st in testpulses['origin'][:]]).nonzero()[0]
                    testpulses.require_dataset('event',
                                               shape=(self.nmbr_channels, len(testpulses['hours']), self.record_length),
                                               dtype=event_dtype)
                    testpulses['event'][:, idx, :] = np.array(pulse_tp, dtype=event_dtype)

                else:
                    testpulses.require_dataset('event', shape=pulse_tp.shape, dtype=event_dtype)
                    testpulses.require_dataset('hours', shape=metainfo_tp[0, :, 10].shape, dtype=float)
                    testpulses.require_dataset('time_s', shape=metainfo_tp[0, :, 4].shape, dtype='int32')
                    testpulses.require_dataset('time_mus', shape=metainfo_tp[0, :, 5].shape, dtype='int32')
                    testpulses['event'][...] = np.array(pulse_tp, dtype=event_dtype)
                    testpulses['hours'][...] = np.array(metainfo_tp[0, :, 10])
                    testpulses['time_s'][...] = np.array(metainfo_tp[0, :, 4], dtype='int32')
                    testpulses['time_mus'][...] = np.array(metainfo_tp[0, :, 5], dtype='int32')

        print('Done.')

    def read_clock_frequency(self, path_rdt, ints_in_header=7, dvm_channels=0):
        """
        Estimate the frequency of the DAQ clock by matching it with the CPU clock, for the last couple of events in
        an RDT file.

        :param path_rdt: Full path to the rdt file.
        :type path_rdt: string
        :param ints_in_header: Either 6 or 7. The correct value is written in the Par file. Recordings with muon veto
            have an additional int (and therefore 7) in the header of each written event.
        :type ints_in_header: int
        :param dvm_channels: The number of DVM channels.
        :type dvm_channels: int
        """

        record = np.dtype([('detector_nmbr', 'i4'),
                           ('coincide_pulses', 'i4'),
                           ('trig_count', 'i4'),
                           ('trig_delay', 'i4'),
                           ('abs_time_s', 'i4'),
                           ('abs_time_mus', 'i4'),
                           ('delay_ch_tp', 'i4', (int(ints_in_header == 7),)),
                           ('time_low', 'i4'),
                           ('time_high', 'i4'),
                           ('qdc_events', 'i4'),
                           ('hours', 'f4'),
                           ('dead_time', 'f4'),
                           ('test_pulse_amplitude', 'f4'),
                           ('dac_output', 'f4'),
                           ('dvm_channels', 'f4', dvm_channels),
                           ('samples', 'i2', self.record_length),
                           ])

        data = np.memmap(path_rdt, dtype=record, mode='r')

        frequencies = (data['time_high'][-10:] * 2 ** 32 + data['time_low'][-10:]) / data[
                                                                                         'hours'][-10:] / 3600

        print('Frequency estimate: {:.1f}'.format(np.mean(frequencies)))

    def include_con_file(self, path_con_file, ints_in_header=7, clock_frequency=None):
        """
        Read the content of a con file an add to HDF5.

        These files contain the control pulse heights and time stamps.

        :param clock_frequency: The frequency of the clock that records the time stamps. This you either know or you
            can read it out from the rdt file, by using with the function read_clock_frequency.
        :type clock_frequency: int
        :param path_con_file: Path to the con file e.g. "data/bcks/*.con".
        :type path_con_file: string
        :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
        :type ints_in_header: int
        """

        print('Accessing CON File...')

        assert self.channels is not None, 'To use this function, you need to specify the channel numbers either in the ' \
                                          'instanciation or when setting the file path for this instance!'

        if clock_frequency is None:
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
                phs = data['pulse_height'][cond]

                # write data to file
                cp_ph[i, ...] = phs[:nmbr_cp]

        print('CON File included.')

    def include_mon(self, path_mon):
        """
        Read the content of an mon file an add to HDF5.

        These files contain the paramters of the cryostat with time stamps.

        :param path_mon: Path to the mon file e.g. "data/bcks/*.mon".
        :type path_mon: string
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

    def include_metainfo(self, path_par):
        """
        Include the metainfo from the PAR file to the HDF5 metainfo group.

        :param path_par: The full path to the PAR file.
        :type path_par: str
        """

        metainfo = get_metainfo(path_par)

        with h5py.File(self.path_h5, 'r+') as f:
            met = f.require_group('metainfo')
            for name in metainfo.keys():
                if name in met:
                    del met[name]
                met.create_dataset(name=name,
                                   data=metainfo[name])

        print('Metainfo included.')

    def include_qdc(self, path_qdc, clock=1e7):
        """
        Read the content of an QDC file an add to HDF5.

        These files contain the hits of the muon veto.

        :param path_qdc: Path to the mon file e.g. "data/bcks/*.qdc".
        :type path_qdc: string
        :param clock: The clock frequency of the recording.
        :type clock: int
        """

        panels = ['cl', 'ltn', 'cr', 'ltf', 'fl', 'lbn', 'fr', 'lbf', 'ftr', 'sum', 'ftl', 'fbr', 'fbl', 'rtf', 'rtn',
                  'rbf', 'rbn', 'btl', 'btr', 'bbl', 'bbr']

        dtype = np.dtype([('time_low', 'uint32'), ('time_high', 'uint32'), ('number_channels', 'int16')] +
                         [(p, 'int16') for p in panels] +
                         [('id{}'.format(i), 'int16') for i, p in enumerate(panels)])

        data = np.fromfile(path_qdc, dtype)

        # create file handles
        with h5py.File(self.path_h5, 'r+') as f:
            qdc = f.require_group('qdc')
            for name in dtype.names:
                if name in qdc:
                    del qdc[name]
                qdc.create_dataset(name=name,
                                   data=data[name])
            if 'hours' in qdc:
                del qdc['hours']
            hours = (data['time_high'] * 2 ** 32 + data['time_low']) / clock / 3600
            qdc.create_dataset(name='hours',
                               data=hours)
            if 'metainfo' in f:
                if 'time_s' in qdc:
                    del qdc['time_s']
                    del qdc['time_mus']

                start_s = f['metainfo']['start_s'][()]
                start_mus = f['metainfo']['start_mus'][()]

                stamp_s = (data['time_high'] * 2 ** 32 + data['time_low']) / clock
                time_s = np.array(stamp_s + start_s + start_mus, dtype=int)
                time_mus = np.array((stamp_s + start_s + start_mus) * 1e6 % 1e6, dtype=int)

                qdc.create_dataset(name='time_s',
                                   data=time_s)
                qdc.create_dataset(name='time_mus',
                                   data=time_mus)
            else:
                print('To include absolute time information, include metainfo first!')

        print('QDC File Included.')
