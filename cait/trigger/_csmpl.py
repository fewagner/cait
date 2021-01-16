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
                  take_samples=-1,  # for all: -1
                  start_time=0,
                  trigger_block=16384,
                  ):
    # TODO

    if overlap is None:
        overlap = int(record_length / 8)

    triggers = []

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
            if block > record_length - overlap:
                block -= record_length - 2 * overlap
                counter += record_length - 2 * overlap
            else:
                # get record window
                record = convert_to_V(stream[counter:counter + record_length])

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
                    triggers.append(sample_to_time(start_time+counter+trig))
                    block += trig + trigger_block
                    if len(triggers) % 100 == 0:
                        print('nmbr triggers: {}, finished: {}'.format(len(triggers), counter/take_samples))

                # increment
                counter += record_length - 2 * overlap
                block -= record_length - 2 * overlap
        # increment
        start_time += (length_stream - 1) * sample_length
    print('#######################################')
    print('DONE WITH ALL FILES FROM THIS CALL.')
    print('Triggers: ', len(triggers))

    return np.array(triggers)

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
               hours=False
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
    if hours:
        time /= 3600

    plt.close()
    plt.plot(time, event)
    plt.show()


def find_nearest(array, value):
    # TODO
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def align_triggers(triggers,  # in seconds
                   trigger_block=16384,  # in samples
                   sample_duration=0.00004,
                   ):
    # TODO

    nmbr_channels = len(triggers)
    active_channels = list(range(nmbr_channels))
    nmbr_triggers = np.array([len(triggers[i]) for i in range(nmbr_channels)])
    aligned_triggers = []
    block_time = trigger_block*sample_duration
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
                       max_time_diff=0.03, # in seconds
                       ):
    # TODO

    flag = np.array([True for i in range(len(trigger_hours))])
    max_hours_diff = max_time_diff/3600

    for val in tp_hours:
        idx = find_nearest(array=trigger_hours, value=val)
        if np.abs(trigger_hours[idx] - val) < max_hours_diff:
            flag[idx] = False

    return flag