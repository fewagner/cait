# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import struct
from ..data._gen_h5 import gen_dataset_from_rdt

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------


class RdtMixin(object):
    """
    Mixin for the DataHandler, includes methods to process the RDT files
    """

    # Checkout RDT File if Channel exists etc
    def checkout_rdt(self, path_rdt, read_events=-1, tpa_list=[0], dvm_channels=0):
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
                    print('measuring hours (0 @ START WRITE): ', hours)
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
                        calc_sev=False, calc_nps=True,
                        processes=4,
                        chunk_size=1000,
                        event_dtype='float64'):
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
                             chunk_size=chunk_size,
                             event_dtype=event_dtype
                             )

        print('Hdf5 dataset created in  {}'.format(path_h5))

        self.path_directory = path_h5

        if self.nmbr_channels == 2:
            self.path_h5 = "{}{}-P_Ch{}-L_Ch{}.h5".format(path_h5, fname, self.channels[0],
                                                          self.channels[1])
            self.fname = fname

        elif self.nmbr_channels > 2:
            path = "{}{}".format(path_h5, fname)
            for i, c in enumerate(self.channels):
                path += '-{}_Ch{}'.format(i + 1, c)
            path += ".h5"
        else:
            raise NotImplementedError(
                'Only for two and more channels implemented!')

        print('Filepath and -name saved.')
