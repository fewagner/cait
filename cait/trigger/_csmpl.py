# imports

import numpy as np
import numba as nb
from ..data._raw import convert_to_V
from ..filter._of import filter_event
from scipy import signal
import matplotlib.pyplot as plt


# functions

def readcs(path):
    """
    This functions reads a continuous stream file i.e. from CRESST
    """
    arr = np.memmap(path, dtype=np.int16, mode='r')
    return arr


@nb.njit
def time_to_sample(t, sample_duration=0.00004):
    # TODO
    s = int(t / sample_duration)
    return s


@nb.njit
def sample_to_time(s, sample_duration=0.00004):
    # TODO
    t = s * sample_duration
    return t


@nb.njit
def time(start_sample, stop_sample, sample_duration=0.00004):
    # TODO
    start_time = start_sample * sample_duration
    stop_time = (stop_sample - 1) * sample_duration
    nmbr_steps = stop_sample - start_sample
    return np.linspace(start_time, stop_time, nmbr_steps)


def trigger_csmpl(paths,
                  trigger_tres,
                  transfer_function=None,
                  record_length=16384,
                  overlap=None,
                  sample_length=0.00004,
                  take_samples=10000000,  # for all: -1
                  start_time=0,
                  trigger_block=16384,
                  ):
    # TODO

    if overlap is None:
        overlap = int(record_length / 8)

    triggers = []
    heights = []

    # global loop for all bck files
    for j, path in enumerate(paths):

        print('#######################################')
        print('CURRENT STREAM NMBR {} PATH {}'.format(j, path))

        stream = readcs(path)
        length_stream = len(stream)
        time = np.arange(start_time, start_time + (length_stream - 1) * sample_length, sample_length)
        start_time += (length_stream - 1) * sample_length
        if take_samples < 0:
            take_samples = length_stream

        # total_nmbr_events = len(stream) / record_length
        print('TOTAL LENGTH STREAM: ', len(stream))

        # ---------------------------------------------------------------
        # TRIGGER ALGO
        # ---------------------------------------------------------------

        counter = 0
        block = 0

        while counter < take_samples - record_length:
            if block > record_length - overlap:
                block -= record_length - 2 * overlap
                counter += record_length - 2 * overlap
            else:
                # get record window
                record = convert_to_V(stream[counter:counter + record_length])
                record_time = time[counter:counter + record_length]

                # remove offset
                record -= np.mean(record[:overlap])

                # filter record window
                if transfer_function is None:
                    filtered_record = signal.medfilt(record, 51)
                else:
                    filtered_record = filter_event(record, transfer_function=transfer_function)

                # trigger
                if block > overlap:
                    trig = np.argmax(filtered_record[block:-overlap] > trigger_tres)
                    if trig > 0:
                        trig += block
                else:
                    trig = np.argmax(filtered_record[overlap:-overlap] > trigger_tres)
                    if trig > 0:
                        trig += overlap
                if trig > 0:
                    triggers.append(record_time[trig])
                    heights.append(filtered_record[trig])
                    block += trig + trigger_block
                    if len(triggers) % 100 == 0:
                        print('nmbr triggers: {}, count: {}/{}'.format(len(triggers), counter, take_samples))

                # increment
                counter += record_length - 2 * overlap
                block -= record_length - 2 * overlap
    print('#######################################')
    print('DONE WITH ALL FILES.')
    print('Triggers: ', len(triggers))

    return np.array(triggers), np.array(heights)

def get_record_window(path,
                      start_time,
                      record_length,
                      sample_duration=0.00004,
                      ):
    # TODO

    bytes_per_sample = 2  # short integer values
    offset = bytes_per_sample * time_to_sample(start_time, sample_duration=0.00004)

    event = np.fromfile(path,
                        offset=offset,
                        count=record_length,
                        dtype=np.short)

    event = convert_to_V(event)
    time = np.arange(start_time, start_time + sample_duration * record_length, sample_duration)

    return event, time


def plot_csmpl(path,
               start_time,
               record_length,
               sample_duration=0.00004,
               ):
    # TODO

    bytes_per_sample = 2  # short integer values
    offset = bytes_per_sample * time_to_sample(start_time, sample_duration=0.00004)

    event = np.fromfile(path,
                        offset=offset,
                        count=record_length,
                        dtype=np.short)

    event = convert_to_V(event)
    time = np.arange(start_time, start_time + sample_duration * record_length, sample_duration)

    plt.close()
    plt.plot(time, event)
    plt.show()

def insort(a, b, kind='mergesort'):
    # TODO
    # took mergesort as it seemed a tiny bit faster for my sorted large array try.
    c = np.concatenate((a, b))  # we still need to do this unfortunatly.
    c.sort(kind=kind)
    flag = np.ones(len(c), dtype=bool)
    np.not_equal(c[1:], c[:-1], out=flag[1:])
    return c[flag]

@nb.njit
def apply_block(arr,
                block,  # arr and block in same units
                ):
    # TODO
    delta_t = np.diff(arr)
    not_blocked = delta_t > block
    arr_new = np.zeros(np.sum(not_blocked) + 1)
    arr_new[0] = arr[0]
    arr_new[1:] = arr[1:][not_blocked]
    return arr_new

@nb.njit
def find_nearest(array, value):
    # TODO
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_close_peak(path,
                   trigger,
                   maximal_deviation,  # in seconds
                   transfer_function,
                   sample_duration=0.00004):
    # TODO
    record_length = (len(transfer_function) - 1) * 2

    event = get_record_window(path=path,
                              start_time=trigger - sample_to_time(record_length / 4),
                              record_length=record_length,
                              sample_duration=0.00004,
                              )

    filtered_event = filter_event(event=event,
                                  transfer_function=transfer_function)

    trigger = time_to_sample(trigger, sample_duration=sample_duration)
    maximal_deviation = time_to_sample(maximal_deviation, sample_duration=sample_duration)

    return np.max(filtered_event[trigger - maximal_deviation: trigger + maximal_deviation])


def align_triggers(triggers,  # in seconds
                   values,  # in V optimum filtered
                   paths,
                   max_channel_diff,  # in time
                   transfer_function,
                   trigger_block=16384,  # in samples
                   sample_duration=0.00004,  # in seconds
                   ):
    # TODO

    nmbr_channels = len(triggers)

    if time_to_sample(max_channel_diff, sample_duration=sample_duration) > len(transfer_function[0]) / 4:
        raise KeyError('max_channel_diff must be smaller than len(transfer_function)/4 !')

    # get all triggers in one array
    c = 1
    while c < nmbr_channels:
        if c == 1:
            aligned_triggers = insort(triggers[0], triggers[1])
        else:
            aligned_triggers = insort(align_triggers, triggers[c])

    # apply the block
    aligned_triggers = apply_block(aligned_triggers,
                                   block=sample_to_time(trigger_block))

    # get corresponding heights
    matched_values = np.zeros([nmbr_channels, len(aligned_triggers)])
    for i, trig in enumerate(aligned_triggers):
        for c in nmbr_channels:
            nearest_idx = find_nearest(array=triggers[c],
                                       value=trig)
            if np.abs(triggers[c][nearest_idx] - trig) < max_channel_diff:
                matched_values[c, i] = values[c][nearest_idx]
            else:
                matched_values[c, i] = get_close_peak(path=paths[c],
                                                      trigger=trig,
                                                      maximal_deviation=max_channel_diff,
                                                      transfer_function=transfer_function[c],
                                                      sample_duration=sample_duration)

    return aligned_triggers, matched_values

def exclude_testpulses(trigger_time,
                       tp_hours,
                       max_time_diff=0.01):
    # TODO

    flag = np.ones(len(trigger_time), dtype=bool)
    tp_hours = 60*60*tp_hours

    for val in tp_hours:
        idx = find_nearest(array=trigger_time, value=val)
        if np.abs(trigger_time[idx] - val) < max_time_diff:
            flag[idx] = False

    return flag