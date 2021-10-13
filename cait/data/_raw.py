# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------

import numpy as np
import numba as nb
from pathlib import Path


# ---------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------

@nb.njit
def convert_to_V(event,
                 bits=16,
                 max=10,
                 min=-10,
                 offset=0):
    """
    Converts an event from int to volt.

    :param event: The event we want to convert.
    :type event: 1D array
    :param bits: Nnumber of bits in each sample.
    :type bits: int,
    :param max: The max volt value.
    :type max: int
    :param min: The min volt value.
    :type min: int
    :param offset: The offset of the volt signal.
    :type offset: int
    :return: The converted event array.
    :rtype: 1D array
    """
    a = 2 ** (bits - 1)
    b = (max - min) / 2 ** bits
    c = min - offset

    event = c + (event + a) * b

    return event


@nb.njit
def convert_to_int(event, bits=16, max=10, min=-10, offset=0):
    """
    Converts an event from volt to int.

    :param event: The event we want to convert.
    :type event: 1D array
    :param bits: Nnumber of bits in each sample.
    :type bits: int,
    :param max: The max volt value.
    :type max: int
    :param min: The min volt value.
    :type min: int
    :param offset: The offset of the volt signal.
    :type offset: int
    :return: The converted event array.
    :rtype: 1D array
    """
    a = 2 ** (bits - 1)
    b = (max - min) / 2 ** bits
    c = min - offset

    event = (event - c) / b - a

    return event


def read_rdt_file(fname, path, channels,
                  remove_offset=False, store_as_int=False, ints_in_header=7, lazy_loading=True):
    """
    Reads a given given hdf5 file and filters out a specific phonon and light
    channel as well as chosen testpulse amplitudes.

    :param fname: Name of the hdf5 file. (with or without the '.rdt' extension)
    :param path: Path to the hdf5 file.
    :param channels: list of ints, the channels from within the rdt file
    :param remove_offset: Removes the offset of an event. (default: False)
    :param store_as_int: bool, if activated the events are saved as 16 bit int instead of 32 bit float
    :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
    :type ints_in_header: int
    :param lazy_loading: Recommended! If true, the data is loaded with memory mapping to avoid memory overflows.
    :type lazy_loading: bool
    :return: returns two arrays of shape (2,n,13) and (2,n,m), where the first
            one contains the metainformation of the filtered events and the
            second contain the pulses.
    """

    nmbr_channels = len(channels)

    if fname[-4:] == '.rdt':
        fname = fname[:-4]
    # gather dataset information from .par file
    par_file = Path("{}{}.par".format(path, fname))
    if not par_file.is_file():
        raise Exception("No '{}.par' found in directory '{}'.".format(fname, path))
    with open("{}{}.par".format(path, fname), "r") as f:
        for line in f:
            # dvm_channels = 0  # taken from the corresponding *.par file
            if 'DVM channels' in line:
                dvm_channels = int(line.split(' ')[-1])

            # record_length = 16384  # taken from the corresponding *.par file
            if 'Record length' in line:
                record_length = int(line.split(' ')[-1])

            # total number of written records taken from the corresponding *.par file
            if 'Records written' in line:
                nbr_rec_events = int(line.split(' ')[-1])  # total number of recorded events

            # end of needed parameters
            if 'Digitizer Setting' in line:
                break

    record = np.dtype([('detector_nmbr', 'i4'),
                       ('coincide_pulses', 'i4'),
                       ('trig_count', 'i4'),
                       ('trig_delay', 'i4'),
                       ('abs_time_s', 'i4'),
                       ('abs_time_mus', 'i4'),
                       ('delay_ch_tp', 'i4', (int(ints_in_header == 7),)),
                       ('time_low', 'i4'),
                       ('time_high', 'i4'),
                       ('qcd_events', 'i4'),
                       ('hours', 'f4'),
                       ('dead_time', 'f4'),
                       ('test_pulse_amplitude', 'f4'),
                       ('dac_output', 'f4'),
                       ('dvm_channels', 'f4', dvm_channels),
                       ('samples', 'i2', record_length),
                       ])

    if not lazy_loading:
        recs = np.fromfile("{}{}.rdt".format(path, fname), dtype=record)
    else:
        recs = np.memmap("{}{}.rdt".format(path, fname), dtype=record, mode='r')

    # check if all events belong together in the channels
    good_recs = [[] for i in range(nmbr_channels)]

    length_recs = recs.shape[0]
    print('Total Records in File: ', length_recs)

    if nmbr_channels > 1:
        for i in range(length_recs):
            if i > length_recs - nmbr_channels:
                break
            cond = recs[i]['detector_nmbr'] == channels[0]
            for j, c in enumerate(channels[1:]):
                cond = np.logical_and(cond, recs[i + j + 1]['detector_nmbr'] == c)
            if cond:
                for j in range(nmbr_channels):
                    good_recs[j].append(i + j)
    else:  # exceptional handling for case of just one channel
        for i in range(length_recs):
            if recs[i]['detector_nmbr'] == channels[0]:
                good_recs[0].append(i)

    nmbr_good_recs = len(good_recs[0])
    print('Event Counts: ', nmbr_good_recs)

    # write to normal arrays
    metainfo = np.empty([nmbr_channels, nmbr_good_recs, 14], dtype=float)
    dvms = np.empty([nmbr_channels, nmbr_good_recs, dvm_channels], dtype=float)
    if not store_as_int:
        pulse = np.empty([nmbr_channels, nmbr_good_recs, record_length], dtype=float)
    else:
        pulse = np.empty([nmbr_channels, nmbr_good_recs, record_length], dtype='int16')

    for c in range(nmbr_channels):
        for i, d in enumerate(record.descr):
            name = d[0]
            if name == 'delay_ch_tp' and ints_in_header != 7:
                continue
            metainfo[c, :, i] = recs[good_recs[c]][name].reshape(-1)
            if i >= 13:
                break
        dvms[c] = recs[good_recs[c]]['dvm_channels']
        if not store_as_int:
            pulse[c] = convert_to_V(recs[good_recs[c]]['samples'])
        else:
            pulse[c] = recs[good_recs[c]]['samples']

    if remove_offset and not store_as_int:
        pulse = np.subtract(pulse.T, np.mean(pulse[:, :, :int(record_length / 8)], axis=2).T).T

    return metainfo, pulse


def get_metainfo(path_par):
    """
    Read the metainfo from the PAR file.

    :param path_sql: The path of the PAR file.
    :type path_sql: string
    :return: The metadata.
    :rtype: dict
    """

    metainfo = {}

    with open(path_par, 'r') as f:
        for i in range(22):
            line = f.readline()
            if 'Timeofday at start' in line and '[s]' in line:
                metainfo['start_s'] = int(line.split(' ')[-1])
            elif 'Timeofday at start' in line and '[us]' in line:
                metainfo['start_mus'] = int(line.split(' ')[-1])
            elif 'Timeofday at stop' in line and '[s]' in line:
                metainfo['stop_s'] = int(line.split(' ')[-1])
            elif 'Timeofday at stop' in line and '[us]' in line:
                metainfo['stop_mus'] = int(line.split(' ')[-1])
            elif 'Measuring time' in line and '[h]' in line:
                metainfo['runtime'] = float(line.split(' ')[-1])
            elif 'Integers in header' in line:
                metainfo['ints_in_header'] = int(line.split(' ')[-1])
            elif 'Unsigned longs in header' in line:
                metainfo['unsgn_longs_in_header'] = int(line.split(' ')[-1])
            elif 'Reals in header' in line:
                metainfo['reals_in_header'] = int(line.split(' ')[-1])
            elif 'DVM channels' in line:
                metainfo['dvm_channels'] = int(line.split(' ')[-1])
            elif 'Record length' in line:
                metainfo['record_length'] = int(line.split(' ')[-1])
            elif 'Records written' in line:
                metainfo['records_written'] = int(line.split(' ')[-1])
            elif 'First DVM channel' in line:
                metainfo['first_dvm_channel'] = int(line.split(' ')[-1])
            elif 'Pre trigger' in line:
                metainfo['pre_trigger'] = int(line.split(' ')[-1])
            elif 'Time base' in line and '[us]' in line:
                metainfo['time_base_mus'] = int(line.split(' ')[-1])
                metainfo['sample_frequency'] = int(1/metainfo['time_base_mus']*1e6)
            elif 'Trigger mode' in line:
                metainfo['trigger_mode'] = int(line.split(' ')[-1])

    return metainfo