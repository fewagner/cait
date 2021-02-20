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


# functions

def readcs(path):
    """
    This functions reads a continuous stream file i.e. from CRESST
    TODO

    :param path:
    :type path:
    :return:
    :rtype:
    """
    arr = np.memmap(path, dtype=np.int16, mode='r')
    return arr


@nb.njit
def time_to_sample(t, sample_duration=0.00004):
    """
    TODO

    :param t:
    :type t:
    :param sample_duration:
    :type sample_duration:
    :return:
    :rtype:
    """
    s = np.floor(t / sample_duration)
    return s


@nb.njit
def sample_to_time(s, sample_duration=0.00004):
    """
    TODO

    :param s:
    :type s:
    :param sample_duration:
    :type sample_duration:
    :return:
    :rtype:
    """
    t = s * sample_duration
    return t


def time(start_sample, stop_sample, sample_duration=0.00004):
    """
    TODO

    :param start_sample:
    :type start_sample:
    :param stop_sample:
    :type stop_sample:
    :param sample_duration:
    :type sample_duration:
    :return:
    :rtype:
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
                  down=1
                  ):
    """
    TODO

    :param stream:
    :type stream:
    :param counter:
    :type counter:
    :param record_length:
    :type record_length:
    :param overlap:
    :type overlap:
    :param block:
    :type block:
    :param transfer_function:
    :type transfer_function:
    :param down:
    :type down:
    :return:
    :rtype:
    """

    # get record window
    record = convert_to_V(stream[counter:counter + record_length])
    # downsample
    if down > 1:
        record = np.mean(record.reshape((int(len(record) / down), down)), axis=1)
        overlap = int(overlap / down)
        block = int(block / down)

    # remove offset
    record -= np.mean(record[:int(overlap / 2)])
    # filter record window
    if transfer_function is None:
        filtered_record = signal.medfilt(record, 51)
    else:
        filtered_record = filter_event(record, transfer_function=transfer_function)
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
                  start_timestamp=0,
                  transfer_function=None,
                  record_length=16384,
                  overlap=None,
                  sample_length=0.00004,
                  take_samples=-1,  # for all: -1
                  start_hours=0,
                  trigger_block=16384,
                  return_info=False,
                  down=1,
                  ):
    """
    TODO

    :param paths:
    :type paths:
    :param trigger_tres:
    :type trigger_tres:
    :param start_timestamp:
    :type start_timestamp:
    :param transfer_function:
    :type transfer_function:
    :param record_length:
    :type record_length:
    :param overlap:
    :type overlap:
    :param sample_length:
    :type sample_length:
    :param take_samples:
    :type take_samples:
    :param start_hours:
    :type start_hours:
    :param trigger_block:
    :type trigger_block:
    :param down:
    :type down:
    :param return_info:
    :type return_info:
    :return:
    :rtype:
    """

    if overlap is None:
        overlap = int(record_length / 4)

    triggers = []
    if return_info:
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

        # total_nmbr_events = len(stream) / record_length
        print('TOTAL LENGTH STREAM: ', len(stream))

        # ---------------------------------------------------------------
        # TRIGGER ALGO
        # ---------------------------------------------------------------

        counter = np.copy(record_length)
        block = 0

        while counter < take_samples - record_length:
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
                                             )
                if height > trigger_tres:
                    # resample in case higher trigger is in record window
                    counter += (trig - overlap) - 1
                    trig, height = get_max_index(stream=stream,  # memmap array
                                                 counter=counter,  # in samples
                                                 record_length=record_length,
                                                 overlap=overlap,  # in samples
                                                 block=block,  # in samples
                                                 transfer_function=transfer_function,
                                                 down=down,
                                                 )

                    triggers.append(start_hours + sample_to_time(counter + trig))
                    if return_info:
                        trigger_heights.append(height)
                        record_starts.append(start_hours + sample_to_time(counter))
                        blocks.append(block)
                    block += trig + trigger_block
                    if len(triggers) % 100 == 0:
                        print('nmbr triggers: {}, finished: {}'.format(len(triggers), counter / take_samples))

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
                      bytes_per_sample=2  # short integer values
                      ):
    """
    TODO

    :param path:
    :type path:
    :param start_time:
    :type start_time:
    :param record_length:
    :type record_length:
    :param sample_duration:
    :type sample_duration:
    :return:
    :rtype:
    """

    offset = bytes_per_sample * time_to_sample(start_time, sample_duration=0.00004)

    event = np.fromfile(path,
                        offset=int(offset),
                        count=record_length,
                        dtype=np.short)

    event = convert_to_V(event)
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
               dpi=300,
               ):
    """
    TODO

    :param path:
    :type path:
    :param start_time:
    :type start_time:
    :param record_length:
    :type record_length:
    :param end_time:
    :type end_time:
    :param sample_duration:
    :type sample_duration:
    :param hours:
    :type hours:
    :param plot_stamps:
    :type plot_stamps:
    :return:
    :rtype:
    """

    if record_length is None and end_time is None:
        raise KeyError("Either record_length or sample_duration must be specified!")

    bytes_per_sample = 2  # short integer values
    if hours:
        if end_time is not None:
            record_length = int((end_time - start_time) / sample_duration * 3600)
        offset = bytes_per_sample * time_to_sample(start_time * 3600, sample_duration=0.00004)

    else:
        if end_time is not None:
            record_length = int((end_time - start_time) / sample_duration)
        offset = bytes_per_sample * time_to_sample(start_time, sample_duration=0.00004)

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
    plt.plot(time, event)
    if plot_stamps is not None:
        for s in plot_stamps:
            if s > time[0] and s < time[-1]:
                plt.axvline(x=s, color='grey', alpha=0.5)
    make_grid()
    if hours:
        plt.xlabel('Time (h)')
    else:
        plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.show()


def find_nearest(array, value):
    """
    TODO

    :param array:
    :type array:
    :param value:
    :type value:
    :return:
    :rtype:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def align_triggers(triggers,  # in seconds
                   trigger_block=16384,  # in samples
                   sample_duration=0.00004,
                   ):
    """
    TODO

    :param triggers:
    :type triggers:
    :param trigger_block:
    :type trigger_block:
    :param sample_duration:
    :type sample_duration:
    :return:
    :rtype:
    """

    nmbr_channels = len(triggers)
    active_channels = list(range(nmbr_channels))
    nmbr_triggers = np.array([len(triggers[i]) for i in range(nmbr_channels)])
    aligned_triggers = []
    block_time = trigger_block * sample_duration
    idxs = np.zeros(nmbr_channels, dtype=int)

    while active_channels:

        times = [triggers[i][idxs[i]] for i in active_channels]
        aligned_triggers.append(np.min(times))

        # block
        for i in np.copy(active_channels):
            while triggers[i][idxs[i]] <= aligned_triggers[-1] + block_time:
                idxs[i] += 1
                if idxs[i] >= nmbr_triggers[i]:
                    active_channels.remove(i)
                    break

    return np.array(aligned_triggers)


def exclude_testpulses(trigger_hours,
                       tp_hours,
                       max_time_diff=0.03,  # in seconds
                       ):
    """
    TODO

    :param trigger_hours:
    :type trigger_hours:
    :param tp_hours:
    :type tp_hours:
    :param max_time_diff:
    :type max_time_diff:
    :return:
    :rtype:
    """

    flag = np.array([True for i in range(len(trigger_hours))])
    max_hours_diff = max_time_diff / 3600

    for val in tp_hours:
        # TODO something is wroong here
        idx = find_nearest(array=trigger_hours, value=val)
        if np.abs(trigger_hours[idx] - val) < max_hours_diff:
            flag[idx] = False

    return flag


def get_test_stamps(path,
                    channels=None,
                    control_pulses=None,
                    event_rate=int(2.5e4),
                    min_cpa=10.1):
    """
    TODO

    :param path:
    :type path:
    :param channels:
    :type channels:
    :param control_pulses:
    :type control_pulses:
    :param event_rate:
    :type event_rate:
    :return:
    :rtype:
    """

    teststamp = np.dtype([
        ('stamp', np.uint64),
        ('tpa', np.float32),
        ('tpch', np.uint32),
    ])

    stamps = np.fromfile(path, dtype=teststamp)

    hours = stamps['stamp'] / 400 / event_rate / 3600
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


def get_starttime(path_sql, csmpl_channel, csmpl_file_identity):
    """
    TODO

    :param path_sql:
    :type path_sql:
    :param csmpl_channel:
    :type csmpl_channel:
    :param filename:
    :type filename:
    :return: time of file creation in seconds
    :rtype:
    """
    connection = sqlite3.connect(path_sql)
    cursor = connection.cursor()
    sql = "SELECT CREATED FROM FILELIST WHERE ch=? AND LABEL=? AND TYPE=? LIMIT 1"
    adr = (csmpl_channel, csmpl_file_identity, "0",)
    cursor.execute(sql, adr)
    query_results = cursor.fetchone()

    time_created = strptime(query_results[0], '%Y-%m-%d %H:%M:%S')

    return mktime(time_created)
