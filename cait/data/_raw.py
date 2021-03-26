# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------

# import os
import numpy as np
import numba as nb
import struct
# from ..fit._pm_fit import arrays_equal
from pathlib import Path
# import pathlib
# import ipdb

from ._progressBar import printProgressBar


# ---------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------

@nb.njit
def convert_to_V(event, bits=16, max=10, min=-10, offset=0):
    """
    Converts an event from int to volt
    :param event: 1D array of the event
    :param bits: int, number of bits in each sample
    :param max: int, the max volt value
    :param min: int, the min volt value
    :param offset: int, the offset of the volt signal
    :return: 1D array, the converted event array
    """
    a = 2 ** (bits - 1)
    b = (max - min) / 2 ** bits
    c = min - offset

    event = c + (event + a) * b

    return event


@nb.njit
def convert_to_int(event, bits=16, max=10, min=-10, offset=0):
    """
    Converts an event from volt to int
    :param event: 1D array of the event
    :param bits: int, number of bits in each sample
    :param max: int, the max volt value
    :param min: int, the min volt value
    :param offset: int, the offset of the volt signal
    :return: 1D array, the converted event array
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
    :param phonon_channel: For selecting the phonon channel.
    :param light_channel: For selecting the light channel.
    :param tpa_list: List of which testpulse amplitudes are filtered out.
                    (Default: [0.0]; if it contains '1.0' all events are taken)
    :param read_events: Number of events which are read. (default: -1 = read till end)
    :param chunk_size: int, the init size of the arrays, if array full another chunks gets
        allocated
    :param remove_offset: Removes the offset of an event. (default: False)
    :return: returns two arrays of shape (2,n,13) and (2,n,m), where the first
            one contains the metainformation of the filtered events and the
            second contain the pulses.
    """

    nmbr_channels = len(channels)  # this is fixed at the moment, change asap

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

    #with open("{}{}.rdt".format(path, fname), "rb") as f:
        # read whole file
    if not lazy_loading:
        recs = np.fromfile("{}{}.rdt".format(path, fname), dtype=record)
    else:
        recs = np.memmap("{}{}.rdt".format(path, fname), dtype=record, mode='r')

    # check if all events belong together in the channels
    good_recs = [[] for i in range(nmbr_channels)]

    length_recs = len(recs)
    print('Total Records in File: ', length_recs)

    if nmbr_channels > 1:
        for i in range(length_recs):
            if i >= length_recs - nmbr_channels:
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

        # uncomment if needed
        # dvms = np.dstack([p_dvms, l_dvms])

    return metainfo, pulse  # , dvms
