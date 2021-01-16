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

    converted = c + (event + a) * b

    return converted
#
# def read_rdt_file(fname, path, channels,
#                   tpa_list=[0.0], read_events=-1,
#                   chunk_size=1000,
#                   remove_offset=False):
#     """
#     Reads a given given hdf5 file and filters out a specific phonon and light
#     channel as well as chosen testpulse amplitudes.
#     :param fname: Name of the hdf5 file. (with or without the '.rdt' extension)
#     :param path: Path to the hdf5 file.
#     :param phonon_channel: For selecting the phonon channel.
#     :param light_channel: For selecting the light channel.
#     :param tpa_list: List of which testpulse amplitudes are filtered out.
#                     (Default: [0.0]; if it contains '1.0' all events are taken)
#     :param read_events: Number of events which are read. (default: -1 = read till end)
#     :param chunk_size: int, the init size of the arrays, if array full another chunks gets
#         allocated
#     :param remove_offset: Removes the offset of an event. (default: False)
#     :return: returns two arrays of shape (2,n,13) and (2,n,m), where the first
#             one contains the metainformation of the filtered events and the
#             second contain the pulses.
#     """
#
#     nmbr_channels = len(channels) # this is fixed at the moment, change asap
#
#     if fname[-4:] == '.rdt':
#         fname = fname[:-4]
#     # gather dataset information from .par file
#     par_file = Path("{}{}.par".format(path, fname))
#     if not par_file.is_file():
#         raise Exception("No '{}.par' found in directory '{}'.".format(fname, path))
#     with open("{}{}.par".format(path, fname), "r") as f:
#         for line in f:
#             # dvm_channels = 0  # taken from the corresponding *.par file
#             if 'DVM channels' in line:
#                 dvm_channels = int(line.split(' ')[-1])
#
#             # record_length = 16384  # taken from the corresponding *.par file
#             if 'Record length' in line:
#                 record_length = int(line.split(' ')[-1])
#
#             # total number of written records taken from the corresponding *.par file
#             if 'Records written' in line:
#                 nbr_rec_events = int(line.split(' ')[-1])  # total number of recorded events
#
#             # end of needed parameters
#             if 'Digitizer Setting' in line:
#                 break
#
#     if read_events == -1:
#         read_events = nbr_rec_events
#
#     # init arrays with length chunk_size and resize when reached
#     metainfo = np.empty([nmbr_channels, chunk_size, 14], dtype=float)
#     dvms = np.empty([nmbr_channels, chunk_size, dvm_channels], dtype=float)
#     pulse = np.empty([nmbr_channels, chunk_size, record_length], dtype=float)
#
#     with open("{}{}.rdt".format(path, fname), "rb") as f:
#         skip_bytes = 13 * 4 + dvm_channels * 4 + record_length * 2  # 13 nmbr of header bytes
#
#         idx_counter = 0
#         hours_checker = 0
#         read_counter = 0
#         goods_counter = 0
#         buffer = {"dvm": np.empty([nmbr_channels, dvm_channels], dtype=float),
#                   "event": np.empty([nmbr_channels, record_length], dtype=np.short),
#                   "header": np.empty([nmbr_channels, 14], dtype=float)}
#
#         while (read_counter < read_events):
#
#             # print the progress bar
#             printProgressBar(read_counter+1, read_events, prefix = 'Progress:', suffix = 'found: {}'.format(goods_counter), length = 50)
#
#             while idx_counter < nmbr_channels:
#
#                 # read routine
#                 buffer["header"][idx_counter, 0] = struct.unpack('i', f.read(4))[0]  # detector_nmb
#                 read_counter += 1
#
#                 if buffer["header"][idx_counter, 0] in channels:  # if the channel is in the channels
#
#                     # read all header infos of the event
#
#                     buffer["header"][idx_counter, 1] = struct.unpack('i', f.read(4))[0] # coincide_pulses
#                     buffer["header"][idx_counter, 2] = struct.unpack('i', f.read(4))[0] # trig_count
#                     buffer["header"][idx_counter, 3] = struct.unpack('i', f.read(4))[0] # trig_delay
#                     buffer["header"][idx_counter, 4] = struct.unpack('i', f.read(4))[0] # abs_time_s
#                     buffer["header"][idx_counter, 5] = struct.unpack('i', f.read(4))[0] # abs_time_mus
#                     buffer["header"][idx_counter, 6] = struct.unpack('i', f.read(4))[0] # delay_ch_tp
#                     buffer["header"][idx_counter, 7] = struct.unpack('i', f.read(4))[0]  # time_low
#                     buffer["header"][idx_counter, 8] = struct.unpack('i', f.read(4))[0]  # time_high
#                     buffer["header"][idx_counter, 9] = struct.unpack('i', f.read(4))[0]  # qcd_events
#                     buffer["header"][idx_counter, 10] = struct.unpack('f', f.read(4))[0]  # hours
#                     buffer["header"][idx_counter, 11] = struct.unpack('f', f.read(4))[0]  # dead_time
#                     buffer["header"][idx_counter, 12] = struct.unpack('f', f.read(4))[0]  # test_pulse_amplitude
#                     buffer["header"][idx_counter, 13] = struct.unpack('f', f.read(4))[0]  # dac_output
#
#                     # read the dvm channels
#                     for i in range(dvm_channels):
#                         buffer["dvm"][idx_counter, i] = struct.unpack('f', f.read(4))[0]  # 'f'
#
#                     # read the recorded event
#                     for i in range(record_length):
#                         buffer["event"][idx_counter, i] = struct.unpack('h', f.read(2))[0]  # 'h'
#
#                     channel_ok = (buffer["header"][idx_counter, 0] == channels[idx_counter])
#                     hours_ok = ((buffer["header"][idx_counter, 10] == hours_checker) or (hours_checker == 0))
#                     tpa_ok = (buffer["header"][idx_counter, 12] in tpa_list) or (
#                                 (buffer["header"][idx_counter, 12] > 0) and (1.0 in tpa_list))
#
#                     # print(read_counter, channel_ok, hours_ok, tpa_ok)
#
#                     if channel_ok and hours_ok and tpa_ok:
#                         # print('GOT IN, IDX++ ')
#                         hours_checker = buffer["header"][idx_counter, 10]
#                         idx_counter += 1
#                     else:  # reset if the events do not match
#                         hours_checker = 0
#                         idx_counter = 0
#
#                 else:
#                     f.seek(skip_bytes, 1)  # skips the number skip_bytes of bytes
#
#                 if read_counter >= read_events:  # check this condition additionally
#                     if idx_counter < nmbr_channels:
#                         throw_last_one = True
#                     break
#
#             # here the idx_counter exceeded the nmbr_channels
#             # this means we got all channels of one event
#             metainfo[:, goods_counter, :] = buffer["header"]
#             dvms[:, goods_counter, :] = buffer["dvm"]
#             pulse[:, goods_counter, :] = convert_to_V(buffer["event"])
#             # print('GOT out, GOODS++ ')
#             goods_counter += 1
#             idx_counter = 0
#             hours_checker = 0
#
#             if goods_counter >= len(metainfo[0]): # make one chunk longer
#                 metainfo = np.concatenate((metainfo, np.empty([nmbr_channels, chunk_size, 14], dtype=float)), axis=1)
#                 dvms = np.concatenate((dvms, np.empty([nmbr_channels, chunk_size, dvm_channels], dtype=float)), axis=1)
#                 pulse = np.concatenate((pulse, np.empty([nmbr_channels, chunk_size, record_length], dtype=float)), axis=1)
#
#         # check same number of events read on the light and phonon channel
#         # otherwise there will occure problems when using np.dstack
#         # if p_metainfo.shape != l_metainfo.shape:
#         #     min_rd = np.min([p_metainfo.shape[0], l_metainfo.shape[0]])
#         #     p_metainfo = p_metainfo[0:min_rd]
#         #     l_metainfo = l_metainfo[0:min_rd]
#         #     p_pulse = p_pulse[0:min_rd]
#         #     l_pulse = l_pulse[0:min_rd]
#
#         # print(p_metainfo[:, 10])
#         # print(l_metainfo[:, 10])
#
#         # check weather the timestamps of the phonon and light events correspond
#         # if comp_tstamp and not arrays_equal(p_metainfo[:, 10], l_metainfo[:, 10]):
#         #     print("p_metainfo.shape={}\tl_metainfo.shape={}".format(
#         #         p_metainfo.shape, l_metainfo.shape))
#         #     raise Exception("The number of phonon and light events differ.")
#
#     # metainfo = np.array(metainfo)
#     # pulse = np.array(pulse)
#
#     if throw_last_one:
#         goods_counter -= 1
#
#     metainfo = metainfo[:, :goods_counter, :] # last event is malicious
#     pulse = pulse[:, :goods_counter, :]
#     dvms = dvms[:, :goods_counter, :]
#
#     # ipdb.set_trace()
#
#     if remove_offset:
#         pulse = np.subtract(pulse.T, np.mean(pulse[:, :, :int(record_length/8)], axis=2).T).T
#
#         # uncomment if needed
#         # dvms = np.dstack([p_dvms, l_dvms])
#
#     return metainfo, pulse  # , dvms


def read_rdt_file(fname, path, channels,
                  tpa_list=[0.0], read_events=-1,
                  chunk_size=1000,
                  remove_offset=False):
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

    nmbr_channels = len(channels) # this is fixed at the moment, change asap

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
                       ('delay_ch_tp', 'i4'),
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

    with open("{}{}.rdt".format(path, fname), "rb") as f:
        # read whole file
        recs = np.fromfile(f, dtype=record)

        # check if all events belong together in the channels
        good_recs = [[] for i in range(nmbr_channels)]

        length_recs = len(recs)

        for i in range(length_recs):
            if i >= length_recs - nmbr_channels:
                break
            cond = recs[i]['detector_nmbr'] == channels[0]
            for j, c in enumerate(channels[1:]):
                cond = np.logical_and(cond, recs[i+j+1]['detector_nmbr'] == c)
            if cond:
                for j in range(nmbr_channels):
                    good_recs[j].append(i+j)

        nmbr_good_recs = len(good_recs[0])
        print('Event Counts: ', nmbr_good_recs)

        # write to normal arrays
        metainfo = np.empty([nmbr_channels, nmbr_good_recs, 14], dtype=float)
        dvms = np.empty([nmbr_channels, nmbr_good_recs, dvm_channels], dtype=float)
        pulse = np.empty([nmbr_channels, nmbr_good_recs, record_length], dtype=float)

        for c in range(nmbr_channels):
            for i, d in enumerate(record.descr):
                name = d[0]
                metainfo[c, :, i] = recs[good_recs[c]][name]
                if i >= 13:
                    break
            dvms[c] = recs[good_recs[c]]['dvm_channels']
            pulse[c] = convert_to_V(recs[good_recs[c]]['samples'])


    if remove_offset:
        pulse = np.subtract(pulse.T, np.mean(pulse[:, :, :int(record_length/8)], axis=2).T).T

        # uncomment if needed
        # dvms = np.dstack([p_dvms, l_dvms])

    return metainfo, pulse  # , dvms