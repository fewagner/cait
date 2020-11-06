# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------

import os
import numpy as np
import numba as nb
import struct
from ..fit._pm_fit import arrays_equal
from pathlib import Path
import pathlib

from ._progressBar import printProgressBar

# ---------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------

@nb.njit
def convert_to_V(event, bits=16, max=10, min=-10, offset=0):
    a = 2 ** (bits - 1)
    b = (max - min) / 2 ** bits
    c = min - offset

    converted = c + (event + a) * b

    return converted


def read_rdt_file(fname, path, phonon_channel, light_channel,
                  tpa_list=[0.0], read_events=-1, comp_tstamp=True,
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
    :param comp_tstamp:
    :param remove_offset: Removes the offset of an event. (default: False)
    :return: returns two arrays of shape (2,n,13) and (2,n,m), where the first
            one contains the metainformation of the filtered events and the
            secound contain the pulses.
    """
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

    if read_events == -1:
        read_events = nbr_rec_events

    p_metainfo = []
    l_metainfo = []

    # dvm[0:dvm_channels]
    p_dvms = []
    l_dvms = []

    # event[0:record_length-1]
    p_pulse = []
    l_pulse = []

    with open("{}{}.rdt".format(path, fname), "rb") as f:
        skip_bytes = 13 * 4 + dvm_channels * 4 + record_length * 2  # 13 nmbr of header bytes

# printProgressBar(0, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
# for i in range(length):
#       do something ...
#       printProgressBar(i+1, length, prefix = 'Progress:', suffix = 'Complete', length = 50)


        for enbr in range(read_events):
            # if verb:
            printProgressBar(enbr+1, read_events, prefix = 'Reading progress:', suffix = 'Complete', length = 50)
            # if enbr % 5000 == 0:
            #     print('Read {} events.'.format(enbr))

            detector_nbr = struct.unpack('i', f.read(4))[0]

            if (detector_nbr == phonon_channel) or (detector_nbr == light_channel):

                # initialize dvm and event arrays
                dvm = np.empty([dvm_channels], dtype=float)
                event = np.empty([record_length], dtype=np.short)

                # read all header infos of the event

                coincide_pulses = struct.unpack('i', f.read(4))[0]
                trig_count = struct.unpack('i', f.read(4))[0]
                trig_delay = struct.unpack('i', f.read(4))[0]
                abs_time_s = struct.unpack('i', f.read(4))[0]
                abs_time_mus = struct.unpack('i', f.read(4))[0]
                delay_ch_tp = struct.unpack('i', f.read(4))[0]
                time_low = struct.unpack('i', f.read(4))[0]  # 'L'
                time_high = struct.unpack('i', f.read(4))[0]  # 'L'
                qcd_events = struct.unpack('i', f.read(4))[0]  # 'L'
                hours = struct.unpack('f', f.read(4))[0]  # 'f'
                dead_time = struct.unpack('f', f.read(4))[0]  # 'f'
                test_pulse_amplitude = struct.unpack('f', f.read(4))[0]  # 'f'
                dac_output = struct.unpack('f', f.read(4))[0]  # 'f'

                # read the dvm channels
                for i in range(dvm_channels):
                    dvm[i] = struct.unpack('f', f.read(4))[0]  # 'f'

                # read the recorded event
                for i in range(record_length):
                    event[i] = struct.unpack('h', f.read(2))[0]  # 'h'

                if (detector_nbr == phonon_channel) and \
                        ((test_pulse_amplitude in tpa_list) or \
                         ((test_pulse_amplitude > 0) and (1.0 in tpa_list))):
                    # or-expression inlcudes every testpulse
                    p_metainfo.append(np.array([detector_nbr,
                                                coincide_pulses,
                                                trig_count,
                                                trig_delay,
                                                abs_time_s,
                                                abs_time_mus,
                                                delay_ch_tp,
                                                time_low,
                                                time_high,
                                                qcd_events,
                                                hours,
                                                dead_time,
                                                test_pulse_amplitude,
                                                dac_output]))
                    p_dvms.append(np.array(dvm))
                    p_pulse.append(np.array(event))
                elif (detector_nbr == light_channel) and \
                        ((test_pulse_amplitude in tpa_list) or \
                         ((test_pulse_amplitude > 0) and (1.0 in tpa_list))):
                    # or-expression inlcudes every testpulse
                    l_metainfo.append(np.array([detector_nbr,
                                                coincide_pulses,
                                                trig_count,
                                                trig_delay,
                                                abs_time_s,
                                                abs_time_mus,
                                                delay_ch_tp,
                                                time_low,
                                                time_high,
                                                qcd_events,
                                                hours,
                                                dead_time,
                                                test_pulse_amplitude,
                                                dac_output]))
                    l_dvms.append(np.array(dvm))
                    l_pulse.append(np.array(event))

            else:
                f.seek(skip_bytes, 1)
        # print()

        p_metainfo = np.array(p_metainfo)
        l_metainfo = np.array(l_metainfo)

        p_pulse = np.array(p_pulse)
        l_pulse = np.array(l_pulse)

        # uncomment if needed
        # p_dvms = np.array(p_dvms)
        # l_dvms = np.array(l_dvms)

        # convert to volt
        p_pulse = convert_to_V(p_pulse)
        l_pulse = convert_to_V(l_pulse)

        # check same number of events read on the light and phonon channel
        # otherwise there will occure problems when using np.dstack
        if p_metainfo.shape != l_metainfo.shape:
            min_rd = np.min([p_metainfo.shape[0], l_metainfo.shape[0]])
            p_metainfo = p_metainfo[0:min_rd]
            l_metainfo = l_metainfo[0:min_rd]
            p_pulse = p_pulse[0:min_rd]
            l_pulse = l_pulse[0:min_rd]

        # print(p_metainfo)
        # print(l_metainfo)

        # check wether the timestamps of the phonon and light events correspond
        if comp_tstamp and not arrays_equal(p_metainfo[:, 10], l_metainfo[:, 10]):
            print("p_metainfo.shape={}\tl_metainfo.shape={}".format(
                p_metainfo.shape, l_metainfo.shape))
            raise Exception("The number of phonon and light events differ.")

        metainfo = np.array([p_metainfo, l_metainfo])
        pulse = np.array([p_pulse, l_pulse])

        if remove_offset:
            pulse = np.subtract(pulse.T, np.mean(pulse[:, :, :1000], axis=2).T).T

        # uncomment if needed
        # dvms = np.dstack([p_dvms, l_dvms])

    return metainfo, pulse  # , dvms