# imports

import numpy as np
import numba as nb
from ..data._raw import convert_to_V
from ..filter._of import filter_event
from ..styles import use_cait_style, make_grid
from scipy import signal
import matplotlib.pyplot as plt
import sqlite3
from time import time, strptime, mktime
from tqdm.auto import tqdm


# functions

def readcs(path):
    """
    This functions reads a continuous stream file, i.e. from CRESST.

    :param path: Path to the continuos stream file.
    :type path: string
    :return: Return the opened file stream to the memory mapped stream file.
    :rtype: memory mapped array
    """
    arr = np.memmap(path, dtype=np.int16, mode='r')
    return arr


@nb.njit
def time_to_sample(t, sample_duration=0.00004):
    """
    Convert a seconds time stamp to a sample index within the stream.

    :param t: The seconds time stamps.
    :type t: 1D array
    :param sample_duration: The sample duration in second.
    :type sample_duration: float
    :return: The sample indices.
    :rtype: 1D array
    """
    s = np.floor(t / sample_duration)
    return s


@nb.njit
def sample_to_time(s, sample_duration=0.00004):
    """
    Convert a sample index stamp to a seconds time stamp within the stream.

    :param s: The sample indices.
    :type s: 1D array
    :param sample_duration: The sample duration in second.
    :type sample_duration: float
    :return: The seconds time stamps.
    :rtype: 1D array
    """
    t = s * sample_duration
    return t


def time(start_sample, stop_sample, sample_duration=0.00004):
    """
    Return an array of second values of individual samples.

    :param start_sample: The sample to start the array from.
    :type start_sample: int
    :param stop_sample: The last sample in the array.
    :type stop_sample: int
    :param sample_duration: The duration of the samples in the array.
    :type sample_duration: float
    :return: The array of second values.
    :rtype: 1D array
    """
    start_time = start_sample * sample_duration
    stop_time = (stop_sample - 1) * sample_duration
    nmbr_steps = stop_sample - start_sample
    return np.linspace(start_time, stop_time, nmbr_steps)


def get_max_index(stream,  # memmap array
                  counter,  # in samples
                  record_length,
                  overlap,  # in samples
                  block,  # in samples
                  transfer_function,  # if use down, this must already be downsampled
                  down=1,
                  window=True,
                  ):
    """
    Filter a record window from the stream array and return the maximum and maximum position.

    :param stream: The stream array.
    :type stream: 1D array
    :param counter: The first sample of the record window within the stream.
    :type counter: int
    :param record_length: The length of the record window in samples.
    :type record_length: int
    :param overlap: The number of samples that overlap between two record windows that are to be filtered.
    :type overlap: int
    :param block: The trigger block value. Only indices larger than this value can be triggered.
    :type block: int
    :param transfer_function: The transfer function for the filter. If no transfer function is provided, a median filter
        is applied instead.
    :type transfer_function: 1D array of size record_length/2 +1
    :param down: The array gets downsampled by this factor before it gets filtered.
    :type down: int
    :param window: If true, a window function is applied to the record window before filtering. Recommended!
    :type window: bool
    :return: (the trigger index inside the record window, the height of the triggered value)
    :rtype: 2-tuple (int, float)
    """

    # get record window
    record = stream[counter:counter + record_length]

    # downsample
    if down > 1:
        record = np.mean(record.reshape((int(len(record) / down), down)), axis=1)
        overlap = int(overlap / down)
        block = int(block / down)
    record = convert_to_V(record)

    # remove offset
    record -= np.mean(record[:int(overlap / 2)])
    # filter record window
    if transfer_function is None:
        filtered_record = signal.medfilt(record, 51)
    else:
        filtered_record = filter_event(record, transfer_function=transfer_function, window=window)
    # get max
    if block > overlap:
        trig = np.argmax(filtered_record[block:-overlap])
        trig += block
    else:
        trig = np.argmax(filtered_record[overlap:-overlap])
        trig += overlap

    h = filtered_record[trig]

    if down > 1:
        trig *= down

    return int(trig), h


def trigger_csmpl(paths,
                  trigger_tres,
                  transfer_function=None,
                  record_length=16384,
                  overlap=None,
                  sample_length=0.00004,
                  take_samples=-1,  # for all: -1
                  start_hours=0,
                  trigger_block=16384,
                  return_info=False,
                  down=1,
                  window=True,
                  ):
    """
    Trigger a number of CSMPL file of one channel and return the time stamps of all triggers.

    The data format and method was described in "(2018) N. Ferreiro Iachellini, Increasing the sensitivity to
    low mass dark matter in cresst-iii witha new daq and signal processing", doi 10.5282/edoc.23762.

    :param paths: The paths to all CSMPL files. It is not recommended to put more than one path, because this will set
        the time gap in between the files to zero.
    :type paths: list of strings
    :param trigger_tres: The trigger thresholds for all channels.
    :type trigger_tres: list of floats
    :param transfer_function: The transfer function for the filter. If no transfer function is provided, a median filter
        is applied instead.
    :type transfer_function: 1D array of size record_length/2 +1
    :param record_length: The length of the record window in samples.
    :type record_length: int
    :param overlap: The number of samples that overlap between two record windows that are to be filtered.
    :type overlap: int
    :param sample_length: The sample length in seconds. If None, it is calculated from the sample frequency.
        :type sample_length: float
    :param take_samples: The number of samples, counted from the start of the of the stream, to trigger. If -1, take
        all samples.
    :type take_samples: int
    :param start_hours: An hours value that is added to all trigger time stamps.
    :type start_hours: float
    :param trigger_block: The first trigger_block samples cannot get triggered.
    :type trigger_block: int
    :param down: The array gets downsampled by this factor before it gets filtered.
    :type down: int
    :param return_info: If true, instead of only the trigger time stamps a tuple is return. The first entry in the tuple
        are the trigger time stamps, the second the trigger heights, third the start values of the record windows,
        fourth the trigger block values of the individual trigger windows.
    :type return_info: bool
    :param window: If true, a window function is applied to the record window before filtering. Recommended!
    :type window: bool
    :return: The hours time stamps of all triggers.
    :rtype: 1D array
    """

    if overlap is None:
        overlap = int(record_length / 8)
    else:
        overlap = int(record_length * overlap)

    triggers = []
    trigger_heights = []
    record_starts = []
    blocks = []

    # global loop for all bck files
    for j, path in enumerate(paths):

        print('#######################################')
        print('CURRENT STREAM NMBR {} PATH {}'.format(j, path))

        stream = readcs(path)
        length_stream = len(stream)

        if take_samples < 0:
            take_samples = length_stream

        print('TOTAL LENGTH STREAM: ', length_stream)

        # ---------------------------------------------------------------
        # TRIGGER ALGO
        # ---------------------------------------------------------------

        with tqdm(total=take_samples - record_length) as pbar:
            pbar.update(record_length)
            counter = np.copy(record_length)
            block = 0
            while counter < take_samples - record_length:
                pbar.update(record_length - 2 * overlap)
                if block >= record_length - overlap:
                    block -= record_length - 2 * overlap
                    counter += record_length - 2 * overlap
                else:
                    trig, height = get_max_index(stream=stream,  # memmap array
                                                 counter=counter,  # in samples
                                                 record_length=record_length,
                                                 overlap=overlap,  # in samples
                                                 block=block,  # in samples
                                                 transfer_function=transfer_function,
                                                 down=down,
                                                 window=window,
                                                 )
                    if height > trigger_tres:
                        # resample in case higher trigger is in record window
                        counter += (trig - overlap) - 1
                        pbar.update((trig - overlap) - 1)
                        trig, height = get_max_index(stream=stream,  # memmap array
                                                     counter=counter,  # in samples
                                                     record_length=record_length,
                                                     overlap=overlap,  # in samples
                                                     block=block,  # in samples
                                                     transfer_function=transfer_function,
                                                     down=down,
                                                     window=window,
                                                     )
                        if height > trigger_tres:
                            triggers.append(start_hours + sample_to_time(counter + trig, sample_duration=sample_length))
                            trigger_heights.append(height)
                            record_starts.append(start_hours + sample_to_time(counter, sample_duration=sample_length))
                            blocks.append(block)

                            block += trig + trigger_block

                    # increment
                    counter += record_length - 2 * overlap
                    block -= record_length - 2 * overlap
                    if block < 0:
                        block = 0
        # increment
        start_hours += (length_stream - 1) * sample_length
    print('#######################################')
    print('DONE WITH ALL FILES FROM THIS CALL.')
    print('Triggers: ', len(triggers))

    if return_info:
        return np.array(triggers), np.array(trigger_heights), np.array(record_starts), np.array(blocks)
    else:
        return np.array(triggers)


def get_record_window(path,
                      start_time,  # in s
                      record_length,
                      sample_duration=0.00004,
                      down=1,
                      bytes_per_sample=2,  # short integer values
                      ):
    """
    Get a record window from a stream *.csmpl file.

    :param path: The path to the *.csmpl file.
    :type path: string
    :param start_time: The start time of the record window from beginning of the file, in seconds.
    :type start_time: float
    :param record_length: The length of the record window in samples.
    :type record_length: int
    :param sample_duration: The duration of the samples in the array.
    :type sample_duration: float
    :param down: The array gets downsampled by this factor before it gets filtered.
    :type down: int
    :param bytes_per_sample:
    :type bytes_per_sample:
    :return: (the values of the record window, the time stamps of the individual samples)
    :rtype: 2-tuple of 1D arrays
    """

    offset = bytes_per_sample * time_to_sample(start_time, sample_duration=sample_duration)
    offset = np.maximum(offset, 0)

    event = np.fromfile(path,
                        offset=int(offset),
                        count=record_length,
                        dtype=np.short)

    event = convert_to_V(event)

    # handling end of file and fill up with small random values to avoid division by zero
    if len(event) < record_length:
        new_event = np.random.normal(scale=1e-5, size=record_length)
        new_event[:len(event)] = event
        event = np.copy(new_event)
        del new_event

    if down > 1:
        event = np.mean(event.reshape(int(len(event) / down), down), axis=1)
    time = start_time + np.arange(0, record_length / down) * sample_duration * down

    return event, time


def plot_csmpl(path,
               start_time=0,
               record_length=None,
               end_time=None,
               sample_duration=0.00004,
               hours=False,
               plot_stamps=None,
               plot_stamps_second=None,
               dpi=None,
               teststamp_path=None,
               clock=10e6,
               sec_offset=0,
               save_path=None,
               ):
    """
    Plot a part of the stream together with provided trigger time stamps.

    :param path: The path to the *.csmpl file.
    :type path: string
    :param start_time: The start time of the record window from beginning of the file, in seconds.
    :type start_time: float
    :param record_length: The length of the record window in samples.
    :type record_length: int
    :param start_time: The end time of the record window from beginning of the file, in seconds.
    :type start_time: float
    :param sample_duration: The duration of the samples in the array.
    :type sample_duration: float
    :param hours: If true, the plot has hours instead of seconds on the x axis.
    :type hours: bool
    :param plot_stamps: If handed, all of these time stamps are plotted, if they are within the record window. In
        hours. This feature is useful, to debug the trigger algorithm, in case it shows unexpected behaviour.
    :type plot_stamps: float
    :param plot_stamps_second: If handed, all of these time stamps are plotted, if they are within the record window,
        in a different color than the first time stamps. In hours. This feature is useful to compare trigger values of
        different trigger algorithms.
    :type plot_stamps_second: float
    :param dpi: The dots per inch of the plots.
    :type dpi: int
    :param teststamp_path: A path to a *.test_stamp file, these stamps are then plottet instead of the plot_stamps.
    :type teststamp_path: string
    :param clock: The Frequency of the time clock, in Hz. Standard for CRESST is 10MHz.
    :type clock: int
    :param sec_offset: This factor is substracted from the time stamps that are read from the *.test_stamps file.
    :type sec_offset: float
    :param save_path: Save the figure at this path location.
    :type save_path: string
    """

    if record_length is None and end_time is None:
        raise KeyError("Either record_length or sample_duration must be specified!")

    if teststamp_path is not None:
        if plot_stamps is not None:
            raise KeyError("You can not print test stamps from a file and hand additional any!")

        teststamp = np.dtype([
            ('stamp', np.uint64),
            ('tpa', np.float32),
            ('tpch', np.uint32),
        ])

        stamps = np.fromfile(teststamp_path, dtype=teststamp)

        plot_stamps = stamps['stamp'] / clock - sec_offset
        if hours:
            plot_stamps /= 3600

    bytes_per_sample = 2  # short integer values
    if hours:
        if end_time is not None:
            record_length = int((end_time - start_time) / sample_duration * 3600)
        offset = bytes_per_sample * time_to_sample(start_time * 3600, sample_duration=sample_duration)

    else:
        if end_time is not None:
            record_length = int((end_time - start_time) / sample_duration)
        offset = bytes_per_sample * time_to_sample(start_time, sample_duration=sample_duration)

    print('Get {} samples from sample {}.'.format(record_length, offset))
    event = np.fromfile(path,
                        offset=int(offset),
                        count=record_length,
                        dtype=np.short)

    event = convert_to_V(event)

    time = np.arange(offset / bytes_per_sample, offset / bytes_per_sample + record_length, 1)
    if hours:
        time = time * sample_duration / 3600
        print('Create time array from {} to {} hours.'.format(time[0], time[-1]))
    else:
        time = time * sample_duration
        print('Create time array from {} to {} seconds.'.format(time[0], time[-1]))

    if len(event) > len(time):
        event = event[:len(time)]
    elif len(time) > len(event):
        time = time[:len(event)]

    print('Plot.')
    plt.close()
    use_cait_style(dpi=dpi)
    plt.plot(time, event, zorder=15)
    if plot_stamps is not None:
        for s in plot_stamps:
            if s > time[0] and s < time[-1]:
                plt.axvline(x=s, color='black', alpha=0.6, linewidth=3.5)
    if plot_stamps_second is not None:
        for s in plot_stamps_second:
            if s > time[0] and s < time[-1]:
                plt.axvline(x=s, color='red', alpha=0.4, linewidth=1.5)
    make_grid()
    if hours:
        plt.xlabel('Time (h)')
    else:
        plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def find_nearest(array, value):
    """
    Find the nearest element in an array to a given value.

    :param array: The array.
    :type array: 1D array
    :param value: The value.
    :type value: float
    :return: The array index of the closest element to the value.
    :rtype: int
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def align_triggers(triggers,  # in seconds
                   trigger_block=16384,  # in samples
                   sample_duration=0.00004,
                   tpas=None,
                   ):
    """
    Match the triggers from multiple csmpl file.

    This function sorts all triggers from different channels into one array and introduces a trigger block
    after each trigger value.

    :param triggers: A list of all triggers from the different channels.
    :type triggers: list of 1D array
    :param trigger_block: A number of samples for which the trigger is blocked after a triggered sample. Typically this
        is the same value as the record window length. This value is the smallest, that rejects overlap between record
        windows.
    :type trigger_block: int
    :param sample_duration: The length of a sample in seconds.
    :type sample_duration: float
    :param tpas:
    :type tpas:
    :return: The aligned triggers. If tpas are provided, the return is a list of the aligned triggers
        and the corresponding tpas.
    :rtype: 1D array
    """

    nmbr_channels = len(triggers)
    active_channels = list(range(nmbr_channels))
    nmbr_triggers = np.array([len(triggers[i]) for i in range(nmbr_channels)])
    aligned_triggers = []
    aligned_tpas = []
    block_time = trigger_block * sample_duration
    idxs = np.zeros(nmbr_channels, dtype=int)

    while active_channels:

        times = [triggers[i][idxs[i]] for i in active_channels]
        aligned_triggers.append(np.min(times))
        if tpas is not None:
            this_tpas = [tpas[i][find_nearest(triggers[i], np.min(times))] for i in range(nmbr_channels)]
            aligned_tpas.append(this_tpas)

        # block
        for i in np.copy(active_channels):
            while triggers[i][idxs[i]] <= aligned_triggers[-1] + block_time:
                idxs[i] += 1
                if idxs[i] >= nmbr_triggers[i]:
                    active_channels.remove(i)
                    break

    if tpas is None:
        return np.array(aligned_triggers)
    else:
        return np.array(aligned_triggers), np.array(aligned_tpas).T


def exclude_testpulses(trigger_hours,
                       tp_hours,
                       max_time_diff=0.5,  # in seconds
                       in_seconds=True,
                       ):
    """
    Exclude all trigger values from the array, that are closer than a minimal distance value to a test pulse time stamp.
    Also all triggers before and after the first and last test pulse are excluded.

    :param trigger_hours: The time stamps of the triggers in hours.
    :type trigger_hours: 1D array
    :param tp_hours: The time stamps of the test pulses in hours.
    :type tp_hours: 1D array
    :param max_time_diff: Trigger values that are closer than this value to any test pulse time stamp get excluded.
        Per default in seconds.
    :type max_time_diff: float
    :param in_seconds: If this is True, the max_time_diff should be handed in seconds. Otherwise, it should be handed
        in hours.
    :type in_seconds: bool
    :return: A array of bool values, that tells which of the trigger time stamps should be kept and which
        excluded.
    :rtype: 1D array
    """

    flag = np.ones(len(trigger_hours), dtype=bool)
    if in_seconds:
        max_hours_diff = max_time_diff / 3600
    else:
        max_hours_diff = max_time_diff  # this is not actually hours then ...

    for val in tp_hours:
        idx = find_nearest(array=trigger_hours, value=val)
        if np.abs(trigger_hours[idx] - val) < max_hours_diff:
            flag[idx] = False

    # also exclude all events after and before tps
    flag[trigger_hours < tp_hours[0]] = 0
    flag[trigger_hours > tp_hours[-1]] = 0

    return flag


def get_test_stamps(path,
                    channels=None,
                    control_pulses=None,
                    clock=10000000,
                    min_cpa=10.1):
    """
    Load the test pulse time stamps from a *.test_stamps file.

    :param path: The path to the *.test_stamps file.
    :type path: string
    :param channels: The test pulse channels we want to read out.
    :type channels: list
    :param control_pulses: If set to True, only control pulses are returned. If False, only test pulses are returned.
        If None, all are returned.
    :type control_pulses: bool or None
    :param clock: The Frequency of the time clock, in Hz. Standard for CRESST is 10MHz.
    :type clock: int
    :return: (the test pulse hours time stamps, the test pulse amplitudes, the channels of the test pulses)
    :rtype: 3-tuple of 1D arrays
    """

    teststamp = np.dtype([
        ('stamp', np.uint64),
        ('tpa', np.float32),
        ('tpch', np.uint32),
    ])

    stamps = np.fromfile(path, dtype=teststamp)

    hours = stamps['stamp'] / clock / 3600
    tpas = stamps['tpa']
    testpulse_channels = stamps['tpch']

    # take only the channels we want
    if channels is not None:
        cond = np.in1d(testpulse_channels, channels)
        hours = hours[cond]
        tpas = tpas[cond]
        testpulse_channels = testpulse_channels[cond]

    # take only control or no control pulses
    if control_pulses is not None:
        if control_pulses:
            cond = tpas > min_cpa
        else:
            cond = tpas < min_cpa
        hours = hours[cond]
        tpas = tpas[cond]
        testpulse_channels = testpulse_channels[cond]

    return hours, tpas, testpulse_channels


def get_starttime(path_sql, csmpl_channel, sql_file_label):
    """
    Read the start time of a *.csmpl file from the SQL database.

    Attention, the start time is only in seconds. This produces an error of the absolute time stamp of up to one
    second.

    :param path_sql: The path of the SQL file.
    :type path_sql: string
    :param csmpl_channel: The channel number of the *.csmpl file. This is either contained in the file name of the
        csmpl file or can be looked up in the SQL database.
    :type csmpl_channel: string
    :param sql_file_label: The file label of the *.csmpl file within SQL database, e.g. bck_001.
    :type sql_file_label: string
    :return: Time of file creation in seconds.
    :rtype: float
    """
    connection = sqlite3.connect(path_sql)
    cursor = connection.cursor()
    sql = "SELECT CREATED FROM FILELIST WHERE ch=? AND LABEL=? AND TYPE=? LIMIT 1"
    adr = (csmpl_channel, sql_file_label, "0",)
    cursor.execute(sql, adr)
    query_results = cursor.fetchone()

    time_created = strptime(query_results[0], '%Y-%m-%d %H:%M:%S')

    return mktime(time_created)


def get_offset(path_dig_stamps):
    """
    Get the offset between start of the continuous DAQ and start of the CCS time recording.

    :param path_dig_stamps: The full path to the *.dig file.
    :type path_dig_stamps: str
    :return: The offset that needs to be subtracted from all CCS time stamps, to get the time stamps w.r.t. the start
        of the CSMPL file.
    :rtype: int
    """

    dig = np.dtype([
        ('stamp', np.uint64),
        ('bank', np.uint32),
        ('bank2', np.uint32),
    ])

    diq_stamps = np.fromfile(path_dig_stamps, dtype=dig)
    dig_samples = diq_stamps['stamp']
    offset_clock = (dig_samples[1] - 2 * dig_samples[0])

    return offset_clock
