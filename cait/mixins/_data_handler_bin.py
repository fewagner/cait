# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from ..features._mp import calc_main_parameters
from ..trigger._csmpl import trigger_csmpl, align_triggers, sample_to_time, \
    exclude_testpulses, get_starttime, get_test_stamps, get_offset
from ..trigger._bin import get_record_window_vdaq, trigger_bin
from tqdm.auto import tqdm
from ..fit._pm_fit import fit_pulse_shape
from ..fit._templates import pulse_template
import os


# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class BinMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for the triggering of *.bin files.
    """

    def include_vtrigger_stamps(self,
                                triggers: list,  # in seconds
                                name_appendix: str = '',
                                trigger_block: int = None,
                                file_start: float = None,  # in seconds
                                ):
        """
        Include trigger time stamps corresponding to a *.bin file. This function does not do the triggering itself,
        has to be done previously, with third-party software, e.g. https://github.com/fewagner/trigger.

        :param triggers: A list of the triggers, corresponding to all channels, in seconds from start of file.
        :type triggers: list of nmbr_channels 1D numpy arrays
        :param name_appendix: This is appended to the name of the trigger data set in the HDF5 file.
        :type name_appendix: str
        :param trigger_block: The number of samples for which the trigger is blocked, after a previous trigger.
        :type trigger_block: int
        :param file_start: This value is added to all triggers, in seconds. Can e.g. be the absolut linux time stamp of
            the file start.
        :type file_start: float
        """

        if trigger_block is None:
            trigger_block = self.record_length

        # fix the number of triggers
        if len(triggers) > 1:
            print('ALIGN TRIGGERS')
            aligned_triggers = align_triggers(triggers=triggers,  # in seconds
                                              trigger_block=trigger_block,
                                              sample_duration=1 / self.sample_frequency,
                                              )
        else:
            aligned_triggers = triggers[0]

        with h5py.File(self.path_h5, 'a') as h5f:

            stream = h5f.require_group(name='stream')
            # write them to file
            print('ADD DATASETS TO HDF5')
            if "trigger_hours{}".format(name_appendix) in stream:
                print('overwrite old trigger_hours{}'.format(name_appendix))
                del stream['trigger_hours{}'.format(name_appendix)]
            stream.create_dataset(name='trigger_hours' + name_appendix,
                                  data=aligned_triggers / 3600)

            if file_start is not None:
                # get file handle
                if "trigger_time_s{}".format(name_appendix) in stream:
                    print('overwrite old trigger_time_s{}'.format(name_appendix))
                    del stream['trigger_time_s{}'.format(name_appendix)]
                stream.create_dataset(name='trigger_time_s' + name_appendix,
                                      shape=(aligned_triggers.shape),
                                      dtype=int)
                if "trigger_time_mus{}".format(name_appendix) in stream:
                    print('overwrite old trigger_time_mus{}'.format(name_appendix))
                    del stream['trigger_time_mus{}'.format(name_appendix)]
                stream.create_dataset(name='trigger_time_mus{}'.format(name_appendix),
                                      shape=(aligned_triggers.shape),
                                      dtype=int)

                stream['trigger_time_s{}'.format(name_appendix)][...] = np.array(aligned_triggers + file_start,
                                                                                 dtype='int32')
                stream['trigger_time_mus{}'.format(name_appendix)][...] = np.array(1e6 * (
                        aligned_triggers + file_start - np.floor(aligned_triggers + file_start)), dtype='int32')

            print('DONE')

    def include_bin_triggers(self,
                             path: str,
                             dtype: np.dtype,
                             keys: list,
                             header_size: int,
                             thresholds: list,  # in V
                             adc_bits=16,
                             trigger_block: int = None,
                             take_samples: int = None,
                             of: list = None,
                             down: int = 1,
                             window: bool = True,
                             overlap: float = None,
                             read_triggerstamps: bool = False,
                             ):
        """
        Trigger the *.bin file of a detector module and include them in the HDF5 set.

        The trigger time stamps of all channels get aligned, by applying a trigger block to channels that belong to the
        same module.

        :param path: The full path for the BIN file.
        :type path: str
        :param dtype: The data type with which we read the *.bin file.
        :type dtype: numpy data type
        :param key: The key of the dtype, corresponding to the channel that we want to read.
        :type key: str
        :param header_size: The size of the file header of the bin file, in bytes.
        :type header_size: int
        :param thresholds: The trigger tresholds for all channels in Volt.
        :type thresholds: list of floats
        :param adc_bits: The precision of the digitizer.
        :type adc_bits: int
        :param trigger_block: The number of samples for that the trigger is blocked after a trigger. If None, it is
            the length of a record window.
        :type trigger_block: int or None
        :param take_samples: The number of samples that we trigger from the stream file. Standard argument is None, which
            means all samples are triggered.
        :type take_samples: int or None
        :param of: The optimum filter transfer functions for all channels.
        :type of: list of arrays
        :param down: The downsampling factor for triggering.
        :type down: int
        :param window: If true, the trigger window is multiplied with a window function before the filtering.
            This is strongly recommended! Otherwise some artifacts from major differences in the left and right baseline
            level can appear somewhere in the middle of the record window.
        :type window: bool
        :param overlap: A value between 0 and 1 that defines the part of the record window that overlaps with the
            previous/next one. Standard value is 1/4 - it is recommended to use this value!
        :type overlap: float
        :param read_triggerstamps: In case there is already a trigger_hours data set in the HDF5 stream group, we can read
            it instead of doing the triggering again. For this, set this argument to True.
        :type read_triggerstamps: bool
        """

        assert os.path.isfile(path), 'The bin file does not exist!'

        if take_samples is None:
            take_samples = -1

        if trigger_block is None:
            trigger_block = self.record_length

        # open write file
        with h5py.File(self.path_h5, 'a') as h5f:

            stream = h5f.require_group(name='stream')

            if not read_triggerstamps:

                if of is None:
                    print("Read OF Transfer Function from h5 file. Alternatively provide one through the of argument.")
                    of = np.zeros((self.nmbr_channels, int(self.record_length / 2 + 1)), dtype=np.complex)
                    of.real = h5f['optimumfilter']['optimumfilter_real']
                    of.imag = h5f['optimumfilter']['optimumfilter_imag']

                # do the triggering
                time = []
                for c, k in enumerate(keys):
                    print('TRIGGER CHANNEL ', k)
                    # get triggers
                    trig = trigger_bin(paths=[path],
                                         dtype=dtype,
                                         key=k,
                                         header_size=header_size,
                                         trigger_tres=thresholds[c],
                                         bits=adc_bits,
                                         transfer_function=of[c],
                                         take_samples=take_samples,
                                         record_length=self.record_length,
                                         sample_length=1 / self.sample_frequency,
                                         start_hours=0,
                                         trigger_block=trigger_block,
                                         down=down,
                                         window=window,
                                         overlap=overlap,
                                         )

                    time.append(trig)

                # fix the number of triggers
                if len(keys) > 1:
                    print('ALIGN TRIGGERS')
                    aligned_triggers = align_triggers(triggers=time,  # in seconds
                                                      trigger_block=trigger_block,
                                                      sample_duration=1 / self.sample_frequency,
                                                      )
                else:
                    aligned_triggers = time[0]

                # write them to file
                print('ADD DATASETS TO HDF5')
                if "trigger_hours" in stream:
                    print('overwrite old trigger_hours')
                    del stream['trigger_hours']
                stream.create_dataset(name='trigger_hours',
                                      data=aligned_triggers / 3600)

            else:

                aligned_triggers = stream['trigger_hours'][:] * 3600


            # include start time, get file handle
            if "trigger_time_s" in stream:
                print('overwrite old trigger_time_s')
                del stream['trigger_time_s']
            stream.create_dataset(name='trigger_time_s',
                                  shape=aligned_triggers.shape,
                                  dtype=np.int64)
            if "trigger_time_mus" in stream:
                print('overwrite old trigger_time_mus')
                del stream['trigger_time_mus']
            stream.create_dataset(name='trigger_time_mus',
                                  shape=aligned_triggers.shape,
                                  dtype=np.int64)

            file_start_s = np.fromfile(path, dtype=dtype, count=1, offset=header_size)['Time'][0]/1e9

            stream['trigger_time_s'][...] = np.array(
                aligned_triggers + file_start_s, dtype='int32')
            stream['trigger_time_mus'][...] = np.array(1e6 * (
                    aligned_triggers + file_start_s -
                    np.floor(aligned_triggers + file_start_s)), dtype='int32')

            print('DONE')


    def include_triggered_events_vdaq(self,
                                      path,
                                      dtype,
                                      keys,
                                      header_size,
                                      adc_bits=16,
                                      max_time_diff=0.5,  # in sec
                                      exclude_tp=True,
                                      sample_duration=None,
                                      name_appendix='',
                                      datatype='float32',
                                      min_tpa=None,
                                      min_cpa=None,
                                      down=1,
                                      origin=None):
        """
        Include the triggered events from the BIN file.

        It is recommended to exclude the testpulses. This means, we exclude them from the events group and put them
        separately in the testpulses and control pulses groups.

        :param path: The full path to the *.bin file.
        :type path: str
        :param dtype: The data type with which we read the *.bin file.
        :type dtype: numpy data type
        :param keys: The keys of the dtype, corresponding to the ADC channels that we want to include.
        :type keys: str
        :param header_size: The size of the file header of the bin file, in bytes.
        :type header_size: int
        :param adc_bits: The precision of the digitizer.
        :type adc_bits: int
        :param max_time_diff: The maximal time difference between a trigger and a test pulse time stamp such that the
            trigger is still counted as test pulse, in seconds.
        :type max_time_diff: float
        :param exclude_tp: If true, we separate the test pulses from the triggered events and put them in individual
            groups.
        :type exclude_tp: bool
        :param sample_duration: The duration of a sample in seconds. If None, the inverse of the sample frequency is taken.
        :type sample_duration: float
        :param name_appendix: The name appendix of the data sets trigger_hours, trigger_time_s and trigger_time_mus in
            the HDF5 file. This is typically needed, when we want to include the events corresponding to time stamps
            that were triggered with the CAT CTrigger and then included in the HDF5 set.
        :type name_appendix: string
        :param datatype: The datatype of the stored events.
        :type datatype: string
        :param min_tpa: TPA values below this are not counted as testpulses.
        :type min_tpa: float
        :param min_cpa: TPA values above this are counted as control pulses.
        :type min_cpa: float
        :param down: The events get stored downsampled by this factor.
        :type down: int
        :param origin: The name of the bin file from which we read, e.g. bck_xxx
        :type origin: str
        """

        if min_tpa is None:
            min_tpa = [0.0001 for i in range(self.nmbr_channels)]
        if min_cpa is None:
            min_cpa = [10.1 for i in range(self.nmbr_channels)]

        if sample_duration is None:
            sample_duration = 1 / self.sample_frequency

        # open read file
        with h5py.File(self.path_h5, 'r+') as h5f:

            stream = h5f['stream']
            trigger_hours = np.array(stream['trigger_hours{}'.format(name_appendix)])
            if "trigger_time_s{}".format(name_appendix) in stream:
                trigger_s = np.array(stream['trigger_time_s{}'.format(name_appendix)])
                trigger_mus = np.array(stream['trigger_time_mus{}'.format(name_appendix)])
            if exclude_tp:
                tp_hours = np.array(h5f['stream']['tp_hours'])
                tpas = np.array(h5f['stream']['tpa'])
                if "tp_time_s" in stream:
                    tp_s = np.array(stream['tp_time_s'])
                    tp_mus = np.array(stream['tp_time_mus'])

            # open write file
            write_events = h5f.require_group('events')
            if exclude_tp:
                cond_dac = np.any([tp_hours[tpas[i] >= min_tpa[i]] for i in range(self.nmbr_channels)], axis=0)
                cond_tps = np.all(
                    [np.logical_and(tpas[i] >= min_tpa[i], tpas[i] < min_cpa[i]) for i in range(self.nmbr_channels)],
                    axis=0)
                cond_cps = np.any([tp_hours[tpas[i] >= min_cpa[i]] for i in range(self.nmbr_channels)], axis=0)
                write_testpulses = h5f.require_group('testpulses')
                if origin is None:
                    write_controlpulses = h5f.require_group('controlpulses')
                    if "hours" in write_testpulses:
                        del write_testpulses["hours"]
                    if "testpulseamplitude" in write_testpulses:
                        del write_testpulses["testpulseamplitude"]
                    if "hours" in write_controlpulses:
                        del write_controlpulses["hours"]
                    write_testpulses.create_dataset(name="hours",
                                                    data=tp_hours[cond_tps])
                    write_testpulses.create_dataset(name="testpulseamplitude",
                                                    data=tpas[:, cond_tps])
                    write_controlpulses.create_dataset(name="hours",
                                                       data=tp_hours[cond_cps])
                    if "tp_time_s" in stream:
                        if "time_s" in write_testpulses:
                            del write_testpulses["time_s"]
                        if "time_mus" in write_testpulses:
                            del write_testpulses["time_mus"]
                        if "time_s" in write_controlpulses:
                            del write_controlpulses["time_s"]
                        if "time_mus" in write_controlpulses:
                            del write_controlpulses["time_mus"]
                        write_testpulses.create_dataset(name="time_s",
                                                        data=tp_s[cond_tps])
                        write_testpulses.create_dataset(name="time_mus",
                                                        data=tp_mus[cond_tps])
                        write_controlpulses.create_dataset(name="time_s",
                                                           data=tp_s[cond_cps])
                        write_controlpulses.create_dataset(name="time_mus",
                                                           data=tp_mus[cond_cps])

                print('Exclude Testpulses.')
                flag = exclude_testpulses(trigger_hours=trigger_hours,
                                          tp_hours=tp_hours[cond_dac],
                                          max_time_diff=max_time_diff)

                trigger_hours = trigger_hours[flag]
                if "trigger_time_s{}".format(name_appendix) in stream:
                    trigger_s = trigger_s[flag]
                    trigger_mus = trigger_mus[flag]

            if origin is None:
                if 'hours' in write_events:
                    del write_events['hours']
                write_events.create_dataset(name='hours',
                                            data=trigger_hours)
                if "trigger_time_s{}".format(name_appendix) in stream:
                    if "time_s" in write_events:
                        del write_events['time_s']
                    if "time_mus" in write_events:
                        del write_events['time_mus']
                    write_events.create_dataset(name='time_s',
                                                data=trigger_s)
                    write_events.create_dataset(name='time_mus',
                                                data=trigger_mus)

                if 'event' in write_events:
                    del write_events['event']
                write_events.create_dataset(name='event',
                                            shape=(
                                                self.nmbr_channels, len(trigger_hours), int(self.record_length / down)),
                                            dtype=datatype)
            else:
                write_events.require_dataset(name='event',
                                             shape=(
                                                 self.nmbr_channels, len(trigger_hours),
                                                 int(self.record_length / down)),
                                             dtype=datatype)

            print('Include the triggered events.')
            for c in range(self.nmbr_channels):

                print('Channel ', c)
                if origin is None:
                    iterable = range(len(trigger_hours))
                else:
                    iterable = np.array([st.decode() == origin for st in h5f['events']['origin'][:]]).nonzero()[0]
                for i in tqdm(iterable):
                    write_events['event'][c, i, :], _ = get_record_window_vdaq(path=path,
                                                                               start_time=trigger_hours[
                                                                                              i] * 3600 - sample_to_time(
                                                                                   self.record_length / 4,
                                                                                   sample_duration=sample_duration),
                                                                               record_length=self.record_length,
                                                                               sample_duration=sample_duration,
                                                                               dtype=dtype,
                                                                               key=keys[c],
                                                                               header_size=header_size,
                                                                               bits=adc_bits,
                                                                               down=down)

            if exclude_tp:

                # ------------
                print('Include Testpulse Events.')

                this_hours = tp_hours[cond_tps]

                if origin is None:
                    if 'event' in write_testpulses:
                        del write_testpulses['event']
                    tp_ev = write_testpulses.create_dataset(name='event',
                                                            shape=(self.nmbr_channels, len(this_hours),
                                                                   int(self.record_length / down)),
                                                            dtype=datatype)
                else:
                    tp_ev = write_testpulses.require_dataset(name='event',
                                                             shape=(self.nmbr_channels, len(this_hours),
                                                                    int(self.record_length / down)),
                                                             dtype=datatype)

                for c in range(self.nmbr_channels):

                    print('Channel ', c)
                    if origin is None:
                        iterable = range(len(this_hours))
                    else:
                        iterable = np.array([st.decode() == origin for st in h5f['testpulses']['origin'][:]]).nonzero()[
                            0]
                    for i in tqdm(range(len(iterable))):
                        tp_ev[c, i, :], _ = get_record_window_vdaq(path=path,
                                                                   start_time=this_hours[i] * 3600 - sample_to_time(
                                                                       self.record_length / 4,
                                                                       sample_duration=sample_duration),
                                                                   record_length=self.record_length,
                                                                   sample_duration=sample_duration,
                                                                   dtype=dtype,
                                                                   key=keys[c],
                                                                   header_size=header_size,
                                                                   bits=adc_bits,
                                                                   down=down)

                # ------------
                if origin is None:
                    print('Calculate Control Pulse Heights.')
                    if 'pulse_height' in write_controlpulses:
                        del write_controlpulses['pulse_height']
                    cphs = write_controlpulses.create_dataset(name="pulse_height",
                                                              shape=(self.nmbr_channels, len(tp_hours[cond_cps])),
                                                              dtype=float)
                    this_hours = tp_hours[cond_cps]
                    for c in range(self.nmbr_channels):

                        print('Channel ', c)
                        for i in tqdm(range(len(this_hours))):
                            cp_array, _ = get_record_window_vdaq(path=path,
                                                                 start_time=this_hours[i] * 3600 - sample_to_time(
                                                                     self.record_length / 4,
                                                                     sample_duration=sample_duration),
                                                                 record_length=self.record_length,
                                                                 sample_duration=sample_duration,
                                                                 dtype=dtype,
                                                                 key=keys[c],
                                                                 header_size=header_size,
                                                                 bits=adc_bits,
                                                                 down=down)

                            # subtract offset
                            cp_array -= np.mean(cp_array[:int(len(cp_array) / 8)])

                            # write the heights to file
                            cphs[c, ...] = np.max(cp_array)

            print('DONE')

    def include_dac_triggers(self,
                             path: str,
                             dtype,
                             keys,
                             header_size,
                             thresholds: list,  # in V
                             dac_bits=16,
                             trigger_block: int = None,
                             take_samples: int = None,
                             down: int = 1,
                             window: bool = True,
                             overlap: float = None,
                             ):
        """
        Trigger the DAC channels of a *.bin file of a detector module and include them in the HDF5 set.

        The trigger time stamps of all channels get aligned, by applying a trigger block to channels that belong to the
        same module.

        :param path: The full paths for the bin file.
        :type path: str
        :param dtype: The data type with which we read the *.bin file.
        :type dtype: numpy data type
        :param keys: The keys of the dtype, corresponding to the channels that we want to read.
        :type keys: list of str
        :param header_size: The size of the file header of the bin file, in bytes.
        :type header_size: int
        :param thresholds: The trigger tresholds for all channels in Volt.
        :type thresholds: list of floats
        :param adc_bits: The precision of the digitizer.
        :type adc_bits: int
        :param trigger_block: The number of samples for that the trigger is blocked after a trigger. If None, it is
            the length of a record window.
        :type trigger_block: int or None
        :param take_samples: The number of samples that we trigger from the stream file. Standard argument is None, which
            means all samples are triggered.
        :type take_samples: int or None
        :param down: The downsampling factor for triggering.
        :type down: int
        :param window: If true, the trigger window is multiplied with a window function before the filtering.
            This is strongly recommended! Otherwise some artifacts from major differences in the left and right baseline
            level can appear somewhere in the middle of the record window.
        :type window: bool
        :param overlap: A value between 0 and 1 that defines the part of the record window that overlaps with the
            previous/next one. Standard value is 1/4 - it is recommended to use this value!
        :type overlap: float
        """

        assert os.path.isfile(path), 'The bin file does not exist!'

        if take_samples is None:
            take_samples = -1

        if trigger_block is None:
            trigger_block = self.record_length

        of = [np.ones(int(self.record_length/2 + 1))/self.record_length/2 + 1 for k in keys]

        # do the triggering
        time = []
        heights = []
        for c, k in enumerate(keys):
            print('TRIGGER CHANNEL ', k)
            # get triggers
            trig, height, _, _ = trigger_bin(paths=[path],
                                 dtype=dtype,
                                 key=k,
                                 header_size=header_size,
                                 trigger_tres=thresholds[c],
                                 bits=dac_bits,
                                 transfer_function=of[c],
                                 take_samples=take_samples,
                                 record_length=self.record_length,
                                 sample_length=1 / self.sample_frequency,
                                 start_hours=0,
                                 trigger_block=trigger_block,
                                 down=down,
                                 window=window,
                                 overlap=overlap,
                                 return_info=True,
                                 square=True,
                                 )

            time.append(trig)
            heights.append(height)

        file_start_s = np.fromfile(path, dtype=dtype, count=1, offset=header_size)['Time'][0] / 1e9

        self.include_test_stamps_vdaq(triggers=time,
                                     tpas=heights,
                                     trigger_block=trigger_block,
                                     file_start=file_start_s)

        print('DONE')

    def include_test_stamps_vdaq(self,
                                 triggers: list,  # in seconds
                                 tpas: list,
                                 name_appendix: str = '',
                                 trigger_block: int = None,  # in
                                 file_start: float = None,  # in seconds
                                 ):
        """
        Include the time stamps and TPA from the test pulses on the *.bin file's DAC channels.

        :param triggers: A list of the triggers, corresponding to all channels, in seconds from start of file.
        :type triggers: list of nmbr_channels 1D numpy arrays
        :param tpas: The test pulse amplitudes corresponding to the triggers.
        :type tpas: list of nmbr_channels 1D numpy arrays
        :param name_appendix: This is appended to the name of the trigger data set in the HDF5 file.
        :type name_appendix: str
        :param trigger_block: The number of samples for which the trigger is blocked, after a previous trigger.
        :type trigger_block: int
        :param file_start: This value is added to all triggers, in seconds. Can e.g. be the absolut linux time stamp of
            the file start.
        :type file_start: float
        """

        # align triggers and apply block

        print('ALIGN TRIGGERS')
        aligned_triggers, aligned_tpas = align_triggers(triggers=triggers,  # in seconds
                                                        trigger_block=trigger_block,  # in samples
                                                        sample_duration=1 / self.sample_frequency,
                                                        tpas=tpas,
                                                        )
        hours = aligned_triggers / 3600

        # open file stream
        with h5py.File(self.path_h5, 'a') as h5f:
            h5f.require_group('stream')

            # calc the time stamp seconds and mus

            time_s = np.array(file_start + hours * 3600, dtype='int32')
            time_mus = np.array(1e6 * (file_start + hours * 3600 - np.floor(file_start + hours * 3600)),
                                dtype='int32')

            # write to file

            for set_name in ['tp_hours', 'tpa', 'tp_time_s', 'tp_time_mus']:
                if set_name + name_appendix in h5f['stream']:
                    del h5f['stream'][set_name]
            h5f['stream'].require_dataset(name='tp_hours' + name_appendix,
                                          shape=hours.shape,
                                          dtype=float)
            h5f['stream'].require_dataset(name='tpa' + name_appendix,
                                          shape=aligned_tpas.shape,
                                          dtype=float)
            h5f['stream'].require_dataset(name='tp_time_s' + name_appendix,
                                          shape=hours.shape,
                                          dtype=int)
            h5f['stream'].require_dataset(name='tp_time_mus' + name_appendix,
                                          shape=hours.shape,
                                          dtype=int)

            h5f['stream']['tp_hours' + name_appendix][...] = hours
            h5f['stream']['tpa' + name_appendix][...] = aligned_tpas
            h5f['stream']['tp_time_s' + name_appendix][...] = time_s
            h5f['stream']['tp_time_mus' + name_appendix][...] = time_mus

        print('Test Stamps included.')

    def include_noise_events_vdaq(self,
                                  path,
                                  dtype,
                                  keys,
                                  header_size,
                                  adc_bits=16,
                                  datatype='float32',
                                  origin=None,
                                  down=1,
                                  ):
        """
        Include the events corresponding to chosen noise triggers from the *.bin file.

        :param path: The full path to the *.bin file.
        :type path: str
        :param dtype: The data type with which we read the *.bin file.
        :type dtype: numpy data type
        :param keys: The keys of the dtype, corresponding to the ADC channels that we want to include.
        :type keys: str
        :param header_size: The size of the file header of the bin file, in bytes.
        :type header_size: int
        :param adc_bits: The precision of the digitizer.
        :type adc_bits: int
        :param datatype: The datatype of the stored events.
        :type datatype: string
        :param origin: This is needed in case you want to include events to a dataset that itself is a merge of other
            datasets. Then typically a set of origin strings is included in the set, which specify the origin of the
            individual events within the merged set. By putting a origin string here, the events get included in the
            event data set exactly at the positions where the origin attribute is the same as the handed origin string.
        :type origin: string
        :param down: The factor by which we want to downsample the included events.
        :type down: int
        """

        with h5py.File(self.path_h5, 'r+') as h5f:

            # get the time stamps
            noise_hours = h5f['stream']['noise_hours']
            noise_time_s = h5f['stream']['noise_time_s']
            noise_time_mus = h5f['stream']['noise_time_mus']

            nmbr_all_events = len(noise_hours)

            # make data sets
            noise = h5f.require_group('noise')
            if 'event' in h5f['noise']:
                del h5f['noise']['event']
            noise.require_dataset(name='event',
                                  shape=(self.nmbr_channels, nmbr_all_events,
                                         int(self.record_length / down)),
                                  dtype=datatype)

            if origin is None:
                hours_h5 = noise.require_dataset(name='hours',
                                                 shape=(nmbr_all_events),
                                                 dtype=float)
                time_s_h5 = noise.require_dataset(name='time_s',
                                                  shape=(nmbr_all_events),
                                                  dtype=int)
                time_mus_h5 = noise.require_dataset(name='time_mus',
                                                    shape=(nmbr_all_events),
                                                    dtype=int)
                hours_h5[...] = noise_hours
                time_s_h5[...] = noise_time_s
                time_mus_h5[...] = noise_time_mus

            # get the record windows
            print('Include the triggered events.')
            for c in range(self.nmbr_channels):
                print('Channel ', c)
                if origin is None:
                    iterable = range(nmbr_all_events)
                else:
                    iterable = np.array([st.decode() == origin for st in noise['origin'][:]]).nonzero()[0]
                for i in tqdm(iterable):
                    noise['event'][c, i, :], _ = get_record_window_vdaq(path=path,
                                                                        start_time=noise_hours[
                                                                                       i] * 3600 - sample_to_time(
                                                                            self.record_length / 4,
                                                                            sample_duration=1 / self.sample_frequency),
                                                                        record_length=self.record_length,
                                                                        sample_duration=1 / self.sample_frequency,
                                                                        dtype=dtype,
                                                                        key=keys[c],
                                                                        header_size=header_size,
                                                                        bits=adc_bits,
                                                                        down=down)

            print('Done.')
