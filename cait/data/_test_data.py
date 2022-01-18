# imports

import numpy as np
from ._raw import convert_to_int
from ..fit._templates import pulse_template
import sqlite3
import datetime
import os


# class

class TestData():
    """
    A class for the generation of *.rdt, *.par, *.con, *.csmpl, *.db, *.dig_stamps and *.test_stamps files for
    the testing of all data processing routines.

    :param filepath: The path to the location of the generated data, including file name without appendix,
        e.g. "../data/run01_Test/mock_001".
    :type filepath: string
    :param duration: The duration of the generated measurement time in seconds.
    :type duration: float, > 0
    :param pulser_interval: The interval in which test pulses are sent.
    :type pulser_interval: float, > 0
    :param sample_frequency: The sample frequency of the measurement in Hz.
    :type sample_frequency: int, > 0
    :param channels: A list of the channel numbers, corresponding to channels in RDT or CSMPL files.
    :type channels: list of integers > 0
    :param tpas: A list of the Test Pulse Amplitudes that are sent in this order. A TPA > 10 is set to 10 and
        counted as control pulse. A TPA < 0 is set to 0 and counted as noise baseline recording. A TPA of 0 is
        counted as triggered event and set to the height event_tpa.
    :type tpas: list of floats
    :param event_tpa: The height of a TPA 0 event is sampled from a uniform distribution with with maximum and minimum
        zero, before application of the saturation curve.
    :type event_tpa: list of nmbr_channels floats
    :param baseline_resolution: The standard deviations of the noise, before application of the saturation function.
    :type baseline_resolution: list of nmbr_channels floats
    :param k: The slope parameter of the logistics function, that is used to describe the saturation.
    :type scales: list of nmbr_channels floats > 0
    :param l: The maximal height of the logistics function, that is used to describe the saturation.
    :type slopes: list of nmbr_channels floats > 0
    :param record_length: The number of samples in a record window.
    :type record_length: int > 0, should be power of 2
    :param dvm_channels: The number of DVM channels in the RDT file. This feature is currently not implemented,
        please stick with the standard value of 0.
    :type dvm_channels: just put this to 0
    :param start_s: The linux time stamp in seconds of the start of the measurement.
    :type start_s: int > 0, standard value: 16.10.2020 22:22:00, the time of the first cait commit
    :param offset: The baseline offset of the channels.
    :type offset: list of two floats
    :param fitpar: The parameters of the Proebst-pulse shape for the alternative events of all channels.
    :type fitpar: list of nmbr_channels 1D numpy arrays, containing the 6 fit parameters, consistent with
        the fit_pulse_model
    :param fitpar_carr: The parameters of the Proebst-pulse shape for the events of all channels.
    :type fitpar_carr: list of nmbr_channels 1D numpy arrays, containing the 6 fit parameters, consistent with
        the fit_pulse_model
    :param include_carriers: If true, every second event is a carrier event.
    :type include_carriers: bool
    :param relative_ph_sigma: The relative variation of the pulse height.
    :type relative_ph_sigma: float > 0 and < 1
    :param eventsize: The number of samples per bankswitch in 50kHz.
    :type eventsize: int > 0
    :param samplesdiv: The factor with that eventsize has to be divided, to match it with the sample_frequency.
    :type samplesdiv: int > 0
    :param cdaq_offset: The offset of the cdaq to the hardware daq, in samples with length 1/sample_frequency.
    :type cdaq_offset: int, > 0 but < event_size/samplesdiv
    :param types: If 0 it is a phonon channel, if 1 a light channel. Other types ambiguous.
    :type types: int
    :param clock: The frequency of the clock for the cdaq, for CRESST it is 10MHz.
    :type clock: int > 0

    >>> import cait as ai
    >>> test_data = test_data = ai.data.TestData(filepath='test_001')
    >>> test_data.generate()
    Rdt file written.
    Con file written.
    Par file written.
    Csmpl Files for all Channels written.
    Sql file written.
    Dig_stamps file written.
    Test_stamps file written.
    """

    def __init__(self, filepath: str, duration: float = 92, pulser_interval: float = 3, sample_frequency: int = 25000,
                 channels: list = [0, 1], tpas: list = [20, 0.1, -1., 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10],
                 event_tpa: list = [1, 1], baseline_resolution: list = [0.002, 0.003],
                 slopes: list = [1, 1.5], scales: list = [12, 8], record_length: int = 16384, dvm_channels: int = 0,
                 start_s: int = 1602879720, offset: list = [0, 0],
                 fitpar: list = [[-1.11, 4, 0.02, 4.15, 2.1, 53.06], [0.77, 51.76, 50.81, 37.24, 8.33, 8.59]],
                 fitpar_carrier: list = [[-2.38, 1.73, 1.65, 136, 0.38, 2.13], [0, 0, 0, 1, 1, 1]],
                 include_carriers: bool = True, relative_ph_sigma=0.1,
                 eventsize: int = 2081024, samplesdiv: int = 2, cdaq_offset: int = 30000, types: list = [0, 1],
                 clock: int = 10000000):

        self.filepath = filepath
        self.duration = duration
        self.nmbr_samples = int(duration * sample_frequency)
        self.pulser_interval = pulser_interval
        self.pulser_interval_samples = int(pulser_interval * sample_frequency)
        self.channels = channels
        self.nmbr_channels = len(channels)
        self.nmbr_events = int(self.duration / pulser_interval)
        self.bankswitch_samples = eventsize / samplesdiv
        self.nmbr_bankswitches = int(self.nmbr_samples / self.bankswitch_samples)
        self.scales = scales
        self.slopes = slopes
        self.record_length = record_length
        self.dvm_channels = dvm_channels
        self.fitpar = fitpar
        self.fitpar_carrier = fitpar_carrier
        self.include_carriers = include_carriers
        self.tpas = np.array(tpas)
        self.all_tpas = np.tile(tpas, reps=int(np.ceil(self.nmbr_events / len(tpas))))[:self.nmbr_events]
        self.event_tpa = event_tpa
        self.tpas_nocp = self.tpas[self.tpas <= 10]
        self.baseline_resolution = baseline_resolution
        self.start_s = start_s
        self.sample_frequency = sample_frequency
        self.offset = offset
        self.clock = clock
        self.cdaq_offset = cdaq_offset
        self.eventsize = eventsize
        self.samplesdiv = samplesdiv
        self.types = types
        self.time = (np.arange(0, record_length, dtype=float) - record_length / 4) / self.sample_frequency * 1e3
        self.events = [pulse_template(self.time, *fitpar[0]), pulse_template(self.time, *fitpar[1])]
        self.carrier_events = [pulse_template(self.time, *fitpar_carrier[0]),
                               pulse_template(self.time, *fitpar_carrier[1])]
        self.relative_ph_sigma = relative_ph_sigma
        self.ph_deviations = np.random.normal(loc=1, scale=relative_ph_sigma,
                                              size=self.nmbr_events)
        self.is_carrier = np.random.randint(1 + int(include_carriers), size=self.nmbr_events, dtype=bool)
        self.is_carrier[self.all_tpas > 0.001] = 0
        self.from_source = np.random.randint(2, size=self.nmbr_events, dtype=bool)
        self.from_source[self.is_carrier] = False
        self.event_heights = []
        for tpa in event_tpa:  # this means for all channels
            self.event_heights.append(np.random.exponential(scale=tpa, size=self.nmbr_events))
            self.event_heights[-1][self.is_carrier] /= 2  # carriers are smaller
            self.event_heights[-1][self.from_source] = np.random.normal(loc=tpa, scale=0.1 * tpa, size=int(
                np.sum(self.from_source)))  # these absorbers are from some calibration source
        self._check_filepath()

    def generate(self, start_offset: int = 0, source: bool = None):  # in seconds
        """
        Generate all files from a measurement file (rdt, con, par, csmpl, sql, dig, test).

        Please be careful with the generation and merge of two test data files: You should set the start_offset of the
        second file such, that the record time of both files are well separated (>1 minute). In the process of triggering and
        determination of trigger times the start time of a file is extracted from the time stamps of test pulses and
        might therefore be wrong for intervals of several seconds. As this error is consistently done for all timestamps
        in the second file, it does not influence the analysis - however, if the start_offset of simulated data is too
        close to the end of a previous file, the events will overlap.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        :param source: If this argument is passed, it must be either 'hw' to simulate the files from a hardware data
            aquisition (RDT, PAR, CON) or 'stream' to simulate the files from stream data (CSMPL, SQL, DIG_STAMPS,
            TEST_STAMPS)
        :type source: string or None
        """
        if source == 'hw' or source is None:
            self._generate_rdt_file(start_offset=start_offset)
            self._generate_con_file()
            self._generate_par_file(start_offset=start_offset)
        if source == 'stream' or source is None:
            self._generate_csmpl_files()
            self._generate_sql_file(start_offset=start_offset)
            self._generate_dig_stamps()
            self._generate_test_stamps()
        if source is not None and source != 'hw' and source != 'stream':
            raise KeyError('Argument source must be either hw or stream!')

    def update_duration(self, new_duration: float):
        """
        Update the duration of a measurement file for the next data generation.

        :param new_duration: The new duration in seconds.
        :type new_duration: float
        """

        self.duration = new_duration
        self.nmbr_samples = int(new_duration * self.sample_frequency)
        self.nmbr_events = int(self.duration / self.pulser_interval)
        self.nmbr_bankswitches = int(self.duration / self.bankswitch_samples)

    def update_filepath(self, file_path: str):
        """
        Update the file path of a measurement file for the next generation.

        :param file_path: The new file path.
        :type file_path: string
        """

        self.filepath = file_path
        self._check_filepath()

    # private ---------

    def _check_filepath(self):
        """
        Checks if the directory for the data exists and creates it otherwise.
        """
        directory, file = os.path.split(self.filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _saturation_curve(self, x: float, scale: float, slope: float):
        """
        Logistic function centered to zero.
        """
        return scale * (1 / (1 + np.exp(-slope * x)) - 0.5)

    def _inverse_saturation_curve(self, y: float, scale: float, slope: float):
        """
        The inverse of a logistic function centered to zero.
        """
        return (np.log((scale + 2 * y) / (3 * scale - 2 * y))) ** (1 / slope)

    def _generate_rdt_file(self, start_offset: float = 0):
        """
        Generate an rdt file of the measurement.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """

        record = np.dtype([('detector_nmbr', 'i4'),
                           ('coincide_pulses', 'i4'),
                           ('trig_count', 'i4'),
                           ('trig_delay', 'i4'),
                           ('abs_time_s', 'i4'),
                           ('abs_time_mus', 'i4'),
                           ('delay_ch_tp', 'i4'),
                           ('time_low', 'i4'),
                           ('time_high', 'i4'),
                           ('qcd_events', 'i4'),
                           ('hours', 'f4'),
                           ('dead_time', 'f4'),
                           ('test_pulse_amplitude', 'f4'),
                           ('dac_output', 'f4'),
                           ('dvm_channels', 'f4', 0),
                           ('samples', 'i2', self.record_length),
                           ])

        recs = np.zeros(self.nmbr_events * self.nmbr_channels, dtype=record)

        for e in range(self.nmbr_events):
            for c in range(self.nmbr_channels):
                idx = int(e * self.nmbr_channels + c)
                recs['detector_nmbr'][idx] = self.channels[c]
                recs['coincide_pulses'][idx] = 0
                recs['trig_count'][idx] = e
                recs['trig_delay'][idx] = 0
                recs['abs_time_s'][idx] = int(self.start_s + start_offset + (e + 1) * self.pulser_interval)
                recs['abs_time_mus'][idx] = int(
                    ((self.start_s + start_offset + (e + 1) * self.pulser_interval) % 1) * 1e6)
                recs['delay_ch_tp'][idx] = 0
                recs['time_low'][idx] = int(((e + 1) * self.pulser_interval) / (2 ** 32))
                recs['time_high'][idx] = int(((e + 1) * self.pulser_interval) % (2 ** 32))
                recs['qcd_events'][idx] = 0
                recs['hours'][idx] = (e + 1) * self.pulser_interval / 3600
                recs['dead_time'][idx] = 0
                recs['test_pulse_amplitude'][idx] = self.all_tpas[e]
                recs['dac_output'][idx] = 0

                if self.all_tpas[e] == 0:
                    sample = self.event_heights[c][e]
                else:
                    sample = np.maximum(np.minimum(self.all_tpas[e], 10), 0)
                if self.include_carriers and self.is_carrier[e]:  # carriers
                    sample *= self.carrier_events[c] * self.ph_deviations[e]
                else:
                    sample *= self.events[c] * self.ph_deviations[e]
                sample += np.random.normal(loc=self.offset[c], scale=self.baseline_resolution[c],
                                           size=self.record_length)
                sample = self._saturation_curve(sample, slope=self.slopes[c], scale=self.scales[c])

                recs['samples'][idx][...] = convert_to_int(sample)

        recs = recs[recs['test_pulse_amplitude'] <= 10]

        # write the array to file
        f = open(self.filepath + ".rdt", "bw")
        recs.tofile(f)
        f.close()

        print('Rdt file written.')

    def _generate_con_file(self):
        """
        Generate a con file from the measurement.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """
        record = np.dtype([('detector_nmbr', 'int32'),
                           ('pulse_height', 'float32'),
                           ('time_stamp_low', 'uint32'),
                           ('time_stamp_high', 'uint32'),
                           ('dead_time', 'float32'),
                           ('mus_since_last_tp', 'int32'),
                           ])

        recs = np.zeros(self.nmbr_events * self.nmbr_channels, dtype=record)
        tpas = np.zeros(self.nmbr_events * self.nmbr_channels)

        for e in range(self.nmbr_events):
            for c in range(self.nmbr_channels):
                idx = int(e * self.nmbr_channels + c)
                recs['detector_nmbr'][idx] = self.channels[c]
                recs['pulse_height'][idx] = self._saturation_curve(
                    self.all_tpas[e] * self.ph_deviations[e] + np.random.normal(loc=self.offset[c],
                                                                                scale=
                                                                                self.baseline_resolution[
                                                                                    c], size=1),
                    slope=self.slopes[c],
                    scale=self.scales[c]) * np.random.normal(loc=1,
                                                             scale=0.05, size=1)
                recs['time_stamp_low'][idx] = int((((e + 1) * self.pulser_interval) * 1e7) % (2 ** 32))
                recs['time_stamp_high'][idx] = int(((e + 1) * self.pulser_interval) * 1e7 / (2 ** 32))
                recs['dead_time'][idx] = 0
                recs['mus_since_last_tp'][idx] = 0
                tpas[idx] = self.all_tpas[e]

        recs = recs[tpas > 10]

        # write the array to file
        f = open(self.filepath + ".con", "bw")
        np.zeros(3, dtype=np.int32).tofile(f)
        recs.tofile(f)
        f.close()

        print('Con file written.')

    def _generate_par_file(self, start_offset: float = 0):
        """
        Generate a par file from the measurement.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """

        timestamp_start = datetime.datetime.fromtimestamp(self.start_s + start_offset)
        date_start = timestamp_start.strftime('%Y-%m-%d %H:%M:%S')
        timestamp_stop = datetime.datetime.fromtimestamp(self.start_s + start_offset + self.duration)
        date_stop = timestamp_stop.strftime('%Y-%m-%d %H:%M:%S')

        records_written = np.ones(self.nmbr_events)
        for i in range(self.nmbr_events):
            records_written[i] = self.tpas[int(i % len(self.tpas))] <= 10

        f = open(self.filepath + ".par", "w")
        f.writelines("CRESST Version 1.x\n")
        f.writelines("Timeofday at start  [s]: {}\n".format(int(self.start_s + start_offset)))
        f.writelines("Timeofday at start [us]: {}\n".format(int(((self.start_s + start_offset) % 1) * 1e6)))
        f.writelines("Timeofday at stop   [s]: {}\n".format(int(self.start_s + start_offset + self.duration)))
        f.writelines(
            "Timeofday at stop  [us]: {}\n".format(int(((self.start_s + start_offset + self.duration) % 1) * 1e6)))
        f.writelines("Start writing to file : {}\n".format(date_start))
        f.writelines("Stop writing to file  : {}\n".format(date_stop))
        f.writelines("Measuring time    [h] : {}\n".format(self.duration / 3600))
        f.writelines("\n")
        f.writelines("********************** Record Structure *********************\n")
        f.writelines("Integers in header             : 7\n")
        f.writelines("Unsigned longs in header       : 3\n")
        f.writelines("Reals in header                : 4\n")
        f.writelines("DVM channels                   : 0\n")
        f.writelines("Record length                  : {}\n".format(self.record_length))
        f.writelines("Records written                : {}\n".format(int(np.sum(records_written))))
        f.writelines("First DVM channel              : 0\n")
        f.writelines("\n")
        f.writelines("**********************  Digitizer Setting *******************\n")
        f.writelines("Pre trigger                    : 2\n")
        f.writelines("Time base [us]                 : {}\n".format(int(1/self.sample_frequency*1e6)))
        f.writelines("Trigger mode                   : 5\n")
        f.close()

        print('Par file written.')

    def _generate_csmpl_files(self):
        """
        Generate a csmpl file from the measurement.
        """
        for c in range(self.nmbr_channels):
            # simulate the noise
            arr = np.random.normal(loc=self.offset[c], scale=self.baseline_resolution[c], size=self.nmbr_samples)

            # add all events
            for i in range(self.nmbr_events):
                start_sample = int((i + 1) * self.pulser_interval_samples - (1 / 4) * self.record_length)
                stop_sample = int((i + 1) * self.pulser_interval_samples + (3 / 4) * self.record_length)

                if stop_sample >= self.nmbr_samples:
                    stop_sample = self.nmbr_samples - 1

                duration_event = stop_sample - start_sample

                if self.all_tpas[i] == 0:
                    tpa = self.event_heights[c][i]
                else:
                    tpa = np.maximum(np.minimum(self.all_tpas[i], 10), 0)
                if self.include_carriers and self.is_carrier[i]:
                    arr[start_sample:stop_sample] += tpa * self.carrier_events[c][:duration_event] * self.ph_deviations[
                        i]
                else:
                    arr[start_sample:stop_sample] += tpa * self.events[c][:duration_event] * self.ph_deviations[i]

            # apply the saturation function
            arr = self._saturation_curve(arr, slope=self.slopes[c], scale=self.scales[c])

            # convert to int
            arr_int = np.zeros(self.nmbr_samples, dtype=np.uint16)
            arr_int[:] = convert_to_int(arr)

            # write the array to file
            f = open(self.filepath + "_Ch" + str(c) + ".csmpl", "bw")
            arr_int.tofile(f)
            f.close()

        print('Csmpl Files for all Channels written.')

    def _generate_sql_file(self, start_offset: float = 0):
        """
        Generate an sql file with data of the measurement.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """
        # open sql connection
        connection = sqlite3.connect(self.filepath + ".db")
        cursor = connection.cursor()

        # format date string of start time

        timestamp = datetime.datetime.fromtimestamp(self.start_s + start_offset)
        date = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # create tables
        sql = """ CREATE TABLE IF NOT EXISTS DAQINFO (
                                                    MOD text PRIMARY KEY,
                                                    TRIGGERDELAY integer,
                                                    SAMPLESDIV integer
                                                ); """
        cursor.execute(sql)
        sql = """ CREATE TABLE IF NOT EXISTS FILELIST (
                                                            CH integer,
                                                            FILENAME text PRIMARY KEY,
                                                            TYPE integer,
                                                            LABEL text,
                                                            CREATED text
                                                        ); """
        cursor.execute(sql)

        # write eventsize and samplesdiv
        sql = ''' INSERT OR IGNORE INTO DAQINFO(MOD, TRIGGERDELAY, SAMPLESDIV) VALUES(?,?,?);'''
        adr = ("mod1", self.eventsize, self.samplesdiv)
        cursor.execute(sql, adr)
        connection.commit()

        # write start times
        for c in range(self.nmbr_channels):
            sql = ''' INSERT OR IGNORE INTO FILELIST(CH, FILENAME, TYPE, LABEL, CREATED) VALUES(?,?,?,?,?);'''
            adr = (self.channels[c], self.filepath + "_Ch" + str(self.channels[c]) + ".csmpl", self.types[c],
                   self.filepath.split("/")[-1], date)
            cursor.execute(sql, adr)
            connection.commit()

        print('Sql file written.')

    def _generate_dig_stamps(self):  # start_offset in seconds
        """
        Generate a dig_stamps file of the measurement.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """
        dig = np.dtype([
            ('stamp', np.uint64),
            ('bank', np.uint32),
            ('bank2', np.uint32),
        ])

        stamps = np.empty(self.nmbr_bankswitches, dtype=dig)
        for b in range(self.nmbr_bankswitches):
            st = (b + 1) * self.bankswitch_samples - self.cdaq_offset  # the dig stamps are in 10MHz samples
            st = np.floor(st * self.clock / self.sample_frequency)
            stamps['stamp'][b] = st
            stamps['bank'][b] = 0
            stamps['bank2'][b] = 0

        f = open(self.filepath + ".dig_stamps", "bw")
        stamps.tofile(f)
        f.close()

        print('Dig_stamps file written.')

    def _generate_test_stamps(self):  # start_offset in seconds
        """
        Generate a test_stamps file of the measurement.

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """
        teststamp = np.dtype([
            ('stamp', np.uint64),
            ('tpa', np.float32),
            ('tpch', np.uint32),
        ])

        stamps = np.empty(self.nmbr_events, dtype=teststamp)
        for e in range(self.nmbr_events):
            stamps['stamp'][e] = (e + 1) * self.pulser_interval_samples - self.cdaq_offset  # the dig stamps are in 10MHz samples
            stamps['stamp'][e] = np.floor(stamps['stamp'][e] * self.clock / self.sample_frequency)
            stamps['tpa'][e] = self.all_tpas[e]
            stamps['tpch'][e] = 0

        f = open(self.filepath + ".test_stamps", "bw")
        stamps.tofile(f)
        f.close()

        print('Test_stamps file written.')
