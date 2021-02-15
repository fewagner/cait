# imports

import numpy as np
from ._raw import convert_to_int
from ..fit._templates import pulse_template
import sqlite3
import datetime


# class

class TestData():
    # needs rdt, con, par

    def __init__(self, filepath, duration=92, pulser_interval=3, sample_frequency=25000,
                 channels=[0, 1], tpas=[20, 0.1, -1., 20, 0.3, 0, 20, 0.5, 1, 20, 3, -1, 20, 0, 10],
                 event_tpa=[0.33, 0.33], baseline_resolution=[0.002, 0.003],
                 k=[1, 1.5], l=[12, 8], record_length=16384, dvm_channels=0, start_s=1602879720, offset=[0, 0],
                 fitpar=[[-1.11, 4, 0.02, 4.15, 2.1, 53.06], [0.77, 51.76, 50.81, 37.24, 8.33, 8.59]],
                 eventsize=2081024, samplesdiv=2, cdaq_offset=30000, types=[0, 1], clock=10000000):
        """
        A class for the generation of test data for all data processing routines.

        :param filepath: The path to the location of the generated data, including file name without appendix,
            e.g. "../data/run01_Test/mock_001"
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
        :param event_tpa: The height of a TPA 0 event, befor application of the saturation curve.
        :type event_tpa: list of nmbr_channels floats
        :param baseline_resolution: The standard deviations of the noise, before application of the saturation function.
        :type baseline_resolution: list of nmbr_channels floats
        :param k: The slope parameter of the logistics function, that is used to describe the saturation.
        :type k: list of nmbr_channels floats > 0
        :param l: The maximal height of the logistics function, that is used to describe the saturation.
        :type l: list of nmbr_channels floats > 0
        :param record_length: The number of samples in a record window.
        :type record_length: int > 0, should be power of 2
        :param dvm_channels: ???
        :type dvm_channels: just put this to 0
        :param start_s: The linux time stamp in seconds of the start of the measurement.
        :type start_s: int > 0, standard value: 16.10.2020 22:22:00, the time of the first cait commit
        :param offset: The baseline offset of the channels.
        :type offset: list of two floats
        :param fitpar: The parameters of the Franz-pulse shape for the events of all channels.
        :type fitpar: list of nmbr_channels 1D numpy arrays, containing the 6 fit parameters, consistent with
            the fit_pulse_model
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
        """
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
        self.k = k
        self.l = l
        self.record_length = record_length
        self.dvm_channels = dvm_channels
        self.fitpar = fitpar
        self.tpas = np.array(tpas)
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

    def generate(self, start_offset=0):  # in seconds
        """
        Generate all files from a measurement file (rdt, con, par, csmpl, sql, dig, test).

        :param start_offset: The time elapsed from start of measurement to start of this file in seconds.
        :type start_offset: float >= 0
        """
        self._generate_rdt_file(start_offset=start_offset)
        self._generate_con_file()
        self._generate_par_file(start_offset=start_offset)
        self._generate_csmpl_files()
        self._generate_sql_file(start_offset=start_offset)
        self._generate_dig_stamps()
        self._generate_test_stamps()

    def update_duration(self, new_duration):
        """
        Update the duration of a measurement file for the next generation.

        :param new_duration: The new duration in seconds.
        :type new_duration: float
        """

        self.duration = new_duration
        self.nmbr_samples = int(new_duration * self.sample_frequency)
        self.nmbr_events = int(self.duration / self.pulser_interval)
        self.nmbr_bankswitches = int(self.duration / self.bankswitch_samples)

    def update_filepath(self, file_path):
        """
        Update the file path of a measurement file for the next generation.

        :param file_path: The new file path.
        :type file_path: string
        """

        self.filepath = file_path

    # private ---------

    def _saturation_curve(self, x, l, k):
        # logistic function centered to zero
        return l * (1 / (1 + np.exp(-k * x)) - 0.5)

    def _inverse_saturation_curve(self, y, l, k):
        # inverse logistic function centered to zero
        return (np.log((l + 2 * y) / (3 * l - 2 * y))) ** (1 / k)

    def _generate_rdt_file(self, start_offset=0):  # in seconds
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
                recs['test_pulse_amplitude'][idx] = self.tpas[int(e % len(self.tpas))]
                recs['dac_output'][idx] = 0

                sample = np.maximum(np.minimum(self.tpas[int(e % len(self.tpas))], 10), 0)
                if sample == 0:
                    sample = self.event_tpa[c]
                sample = sample * self.events[c]
                sample = sample + np.random.normal(loc=self.offset[c], scale=self.baseline_resolution[c],
                                                   size=self.record_length)
                sample = self._saturation_curve(sample, self.l[c], self.k[c])

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
                    self.tpas[int(e % len(self.tpas))] + np.random.normal(loc=self.offset[c],
                                                                          scale=self.baseline_resolution[c], size=1),
                    self.l[c],
                    self.k[c])
                recs['time_stamp_low'][idx] = int((((e + 1) * self.pulser_interval) * 1e7) % (2 ** 32))
                recs['time_stamp_high'][idx] = int(((e + 1) * self.pulser_interval) * 1e7 / (2 ** 32))
                recs['dead_time'][idx] = 0
                recs['mus_since_last_tp'][idx] = 0
                tpas[idx] = self.tpas[int(e % len(self.tpas))]

        recs = recs[tpas > 10]

        # write the array to file
        f = open(self.filepath + ".con", "bw")
        np.zeros(3, dtype=np.int32).tofile(f)
        recs.tofile(f)
        f.close()

        print('Con file written.')

    def _generate_par_file(self, start_offset=0):
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

                tpa = np.maximum(np.minimum(self.tpas[int(i % len(self.tpas))], 10), 0)
                if tpa == 0:
                    tpa = self.event_tpa[c]
                arr[start_sample:stop_sample] += tpa * self.events[c][:duration_event]

            # apply the saturation function
            arr = self._saturation_curve(arr, self.l[c], self.k[c])

            # convert to int
            arr_int = np.zeros(self.nmbr_samples, dtype=np.uint16)
            arr_int[:] = convert_to_int(arr)

            # write the array to file
            f = open(self.filepath + "_Ch" + str(c) + ".csmpl", "bw")
            arr_int.tofile(f)
            f.close()

        print('Csmpl Files for all Channels written.')

    def _generate_sql_file(self, start_offset=0):
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
            st = b * self.bankswitch_samples + np.abs(
                self.bankswitch_samples - self.cdaq_offset)  # the dig stamps are in 10MHz samples
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
            stamps['stamp'][e] = (e + 1) * self.pulser_interval_samples + self.cdaq_offset  # the dig stamps are in 10MHz samples
            stamps['stamp'][e] = np.floor(stamps['stamp'][e] * self.clock / self.sample_frequency)
            stamps['tpa'][e] = self.tpas[int(e % len(self.tpas))]
            stamps['tpch'][e] = 0

        f = open(self.filepath + ".test_stamps", "bw")
        stamps.tofile(f)
        f.close()

        print('Test_stamps file written.')
