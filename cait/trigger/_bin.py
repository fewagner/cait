# imports

import numpy as np
from ..data._raw import convert_to_V
from ._csmpl import time_to_sample, get_max_index, sample_to_time
from tqdm.auto import tqdm
import pdb

# functions

def get_record_window_vdaq(path,
                           start_time,  # in s
                           record_length,
                           dtype,
                           key,
                           header_size,
                           sample_duration=0.00004,
                           down=1,
                           bits=16,
                           vswing=39.3216,
                           ):
    """
    Get a record window from a stream *.bin file.

    :param path: The full path of the *.bin file.
    :type path: str
    :param start_time: The start time in seconds, from where we want to read the record window, starting with 0 at
        the beginning of the file.
    :type start_time: float
    :param record_length: The record length to read from the bin file.
    :type record_length: int
    :param dtype: The data type with which we read the *.bin file.
    :type dtype: numpy data type
    :param key: The key of the dtype, corresponding to the channel that we want to read.
    :type key: str
    :param header_size: The size of the file header of the bin file, in bytes.
    :type header_size: int
    :param sample_duration: The duration of a sample, in seconds.
    :type sample_duration: float
    :param down: A factor by which the events are downsampled before they are returned.
    :type down: int
    :param bits: The precision of the digitizer.
    :type bits: int
    :param vswing: The total volt region covered by the ADC.
    :type vswing: float
    :return: List of two 1D numpy arrays: The event read from the *.bin file, and the corresponding time grid.
    :rtype: list
    """

    offset = header_size + dtype.itemsize * time_to_sample(start_time, sample_duration=sample_duration)
    offset = np.maximum(offset, header_size)

    event = np.fromfile(path,
                        offset=int(offset),
                        count=record_length,
                        dtype=dtype)

    event = convert_to_V(event[key], bits=bits, max=vswing / 2, min=-vswing / 2)

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


def bin(s, nmbr_bits=None):
    """
    Returns a string of 0/1 values for any datatype.

    :param s: Any variable or object.
    :type s: any
    :return: The 0/1's of s' bits.
    :rtype: string
    """
    bit_list = str(s) if s <= 1 else bin(s >> 1) + str(s & 1)
    if nmbr_bits is not None:
        while len(bit_list) < nmbr_bits:
            bit_list = '0' + bit_list
    return bit_list


def read_header(path_bin):
    """
    Function that reads the header of a *.bin file.

    :param f: The path to the *.bin file.
    :type f: string
    :return: list (dictionary with infos from header,
                    list of keys that are written in each sample,
                    bool True if adc is 16 bit,
                    bool True if dac is 16 bit)
    :rtype: list
    """

    keys = []

    dt_header = np.dtype([('ID', 'i4'),
                          ('numOfBytes', 'i4'),
                          ('downsamplingFactor', 'i4'),
                          ('channelsAndFormat', 'i4'),
                          ('timestamp', 'uint64'),
                          ])

    header = np.fromfile(path_bin, dtype=dt_header, count=1)[0]

    channelsAndFormat = bin(header['channelsAndFormat'], 32)

    # print('channelsAndFormat: ', channelsAndFormat)

    # bit 0: Timestamp uint64
    if channelsAndFormat[-1] == '1':
        keys.append('Time')

    # bit 1: settings uint32
    if channelsAndFormat[-2] == '1':
        keys.append('Settings')

    # bit 2-5: dac 1-4
    for c, b in enumerate([2, 3, 4, 5]):
        if channelsAndFormat[-int(b + 1)] == '1':
            keys.append('DAC' + str(c + 1))

    # bit 6-8: adc 1-3
    for c, b in enumerate([6, 7, 8]):
        if channelsAndFormat[-int(b + 1)] == '1':
            keys.append('ADC' + str(c + 1))

    # bit 9-16: -

    # bit 16: 0...DAC 16 bit, 1...DAC 32 bit
    if channelsAndFormat[-17] == '1':
        dac_short = False
    else:
        dac_short = True

    # bit 17: 0...ADC 16 bit, 1...ADC 32 bit
    if channelsAndFormat[-18] == '1':
        adc_short = False
    else:
        adc_short = True

    # bit 18-31: -

    # construct data type

    dt_tcp = []

    for k in keys:
        if k.startswith('Time'):
            dt_tcp.append((k, 'uint64'))
        elif k.startswith('Settings'):
            dt_tcp.append((k, 'i4'))
        elif k.startswith('DAC'):
            if dac_short:
                dt_tcp.append((k, 'i2'))
            else:
                dt_tcp.append((k, 'i4'))
        elif k.startswith('ADC'):
            if adc_short:
                dt_tcp.append((k, 'i2'))
            else:
                dt_tcp.append((k, 'i4'))

    dt_tcp = np.dtype(dt_tcp)

    if adc_short:
        adc_bits = 16
    else:
        adc_bits = 24
    if dac_short:
        dac_bits = 16
    else:
        dac_bits = 24

    return header, keys, adc_bits, dac_bits, dt_tcp


def trigger_bin(paths,
                dtype,
                key,
                header_size,
                trigger_tres,
                bits=16,
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
                square=False,
                ):
    """
    Trigger a number of BIN files in one channel and return the time stamps of all triggers.

    :param paths: The paths to all BIN files. It is not recommended to put more than one path, because this will set
        the time gap in between the files to zero.
    :type paths: list of strings
    :param dtype: The data type with which we read the *.bin file.
    :type dtype: numpy data type
    :param key: The key of the dtype, corresponding to the channel that we want to read.
    :type key: str
    :param header_size: The size of the file header of the bin file, in bytes.
    :type header_size: int
    :param trigger_tres: The trigger thresholds for all channels.
    :type trigger_tres: list of floats
    :param bits: The precision of the digitizer.
    :type bits: int
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
    :param return_info: If true, instead of only the trigger time stamps a tuple is return. The first entry in the tuple
        are the trigger time stamps, the second the trigger heights, third the start values of the record windows,
        fourth the trigger block values of the individual trigger windows.
    :type return_info: bool
    :param down: The array gets downsampled by this factor before it gets filtered.
    :type down: int
    :param window: If true, a window function is applied to the record window before filtering. Recommended!
    :type window: bool
    :param square: Square the stream values before triggering, this needs to be done for DAC channels.
    :type square: bool
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

        # stream = readcs(path)
        stream = np.memmap(path, dtype=dtype, mode='r', offset=header_size)[key]
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
                                                 bits=bits,
                                                 max=20,
                                                 min=-20,
                                                 square=square,
                                                 )
                    if height > trigger_tres:
                        # resample in case higher trigger is in record window
                        counter += (trig - overlap) - 1
                        if counter > take_samples - record_length: #check if new record window would end outside sample
                            continue
                        pbar.update((trig - overlap) - 1)
                        trig, height = get_max_index(stream=stream,  # memmap array
                                                     counter=counter,  # in samples
                                                     record_length=record_length,
                                                     overlap=overlap,  # in samples
                                                     block=block,  # in samples
                                                     transfer_function=transfer_function,
                                                     down=down,
                                                     window=window,
                                                     bits=bits,
                                                     max=20,
                                                     min=-20,
                                                     square=square,
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
