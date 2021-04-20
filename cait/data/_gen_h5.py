# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os
from multiprocessing import get_context  # Pool
import h5py
import numpy as np
from ..data._raw import read_rdt_file, convert_to_V
from ..features._mp import calc_main_parameters
from ..fit._pm_fit import fit_pulse_shape
from ._baselines import calculate_mean_nps
from ..fit._sev import generate_standard_event
from ..filter._of import optimal_transfer_function


# import ipdb

# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------


def gen_dataset_from_rdt(path_rdt,
                         fname,
                         path_h5,
                         channels,
                         tpa_list=[0.0],
                         calc_mp=True,
                         calc_fit=False,
                         calc_sev=False,
                         calc_nps=True,
                         processes=4,
                         event_dtype='float32',
                         ints_in_header=7,
                         sample_frequency=25000,
                         lazy_loading=True,
                         ):
    """
    Generates a HDF5 File from an RDT File, optionally MP, Fit, SEV Calculation.

    :param path_rdt: Path to the rdt file e.g. "data/bcks/".
    :type path_rdt: string
    :param fname: Name of the file e.g. "bck_001".
    :type fname: string
    :param path_h5: Path where the h5 file is saved e.g. "data/hdf5s%".
    :type path_h5: string
    :param channels: the numbers of the channels in the hdf5 file that we want to include in rdt
    :type channels: list
    :param tpa_list: The test pulse amplitudes to save, if 1 is in the list, all positive values are included.
    :type tpa_list: list
    :param calc_mp: If True the main parameters for all events are calculated and stored.
    :type calc_mp: bool
    :param calc_fit: Not recommended! If True the parametric fit for all events is calculated and stored.
    :type calc_fit: bool
    :param calc_sev: Not recommended! If True the standard event for all event channels is calculated.
    :type calc_sev: bool
    :param calc_nps: If True the main parameters for all events are calculated and stored.
    :type calc_nps: bool
    :param processes: The number of processes that is used for the code execution.
    :type processes: int
    :param event_dtype: Datatype to save the events with.
    :type event_dtype: string
    :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
    :type ints_in_header: int
    :param sample_frequency: The sample frequency of the records.
    :type sample_frequency: int
    :param lazy_loading: Recommended! If true, the data is loaded with memory mapping to avoid memory overflows.
    :type lazy_loading: bool
    """

    nmbr_channels = len(channels)

    if not os.path.exists(path_h5):
        os.makedirs(path_h5)

    print('READ EVENTS FROM RDT FILE.')

    metainfo, pulse = \
        read_rdt_file(fname=fname,
                      path=path_rdt,
                      channels=channels,
                      store_as_int=False,
                      ints_in_header=ints_in_header,
                      lazy_loading=lazy_loading,
                      )

    # ipdb.set_trace()

    if nmbr_channels == 2:
        path = "{}{}-P_Ch{}-L_Ch{}.h5".format(path_h5, fname,
                                                       channels[0], channels[1])
    else:
        path = "{}{}".format(path_h5, fname)
        for i, c in enumerate(channels):
            path += '-{}_Ch{}'.format(i + 1, c)
        path += ".h5"

    with h5py.File(path, 'w') as h5f:

        for i, c in enumerate(channels):
            h5f.attrs.create('Ch_{}'.format(i + 1), data=c)

        # ################# PROCESS EVENTS #################
        # if we filtered for events
        if 0.0 in tpa_list:

            print('WORKING ON EVENTS WITH TPA = 0.')

            metainfo_event = metainfo[:, metainfo[0, :, 12] == 0, :]
            pulse_event = pulse[:, metainfo[0, :, 12] == 0, :]

            nmbr_events = len(metainfo_event[0])

            events = h5f.create_group('events')
            events.create_dataset('event', data=np.array(pulse_event, dtype=event_dtype))
            events.create_dataset('hours', data=np.array(metainfo_event[0, :, 10]))
            events.create_dataset('time_s', data=np.array(metainfo_event[0, :, 4]), dtype='int32')
            events.create_dataset('time_mus', data=np.array(metainfo_event[0, :, 5]), dtype='int32')
            print('CREATE DATASET WITH EVENTS.')

            # for small numbers of events only additional overhead is introduced
            processes = 1 if pulse_event.shape[1] < 5 else processes

            if calc_mp:

                print('CALCULATE MAIN PARAMETERS.')

                # 10 is number main parameters
                mainpar_event = np.empty(
                    [nmbr_channels, nmbr_events, 10], dtype=float)

                # basically a for loop running on multiple processes
                with get_context("spawn").Pool(processes) as p:
                    for c in range(nmbr_channels):
                        mainpar_list_event = p.map(
                            calc_main_parameters, pulse_event[c, :, :])
                        mainpar_event[c, :, :] = np.array(
                            [o.getArray() for o in mainpar_list_event])

                events.create_dataset('mainpar', data=np.array(mainpar_event))
                # description of the mainpar (data=col_in_mainpar)
                events['mainpar'].attrs.create(name='pulse_height', data=0)
                events['mainpar'].attrs.create(name='t_zero', data=1)
                events['mainpar'].attrs.create(name='t_rise', data=2)
                events['mainpar'].attrs.create(name='t_max', data=3)
                events['mainpar'].attrs.create(name='t_decaystart', data=4)
                events['mainpar'].attrs.create(name='t_half', data=5)
                events['mainpar'].attrs.create(name='t_end', data=6)
                events['mainpar'].attrs.create(name='offset', data=7)
                events['mainpar'].attrs.create(name='linear_drift', data=8)
                events['mainpar'].attrs.create(name='quadratic_drift', data=9)

            if calc_fit:

                print('CALCULATE FIT.')

                # 6 is number fit parameters
                fitpar_event = np.empty(
                    [nmbr_channels, nmbr_events, 6], dtype=float)

                with get_context("spawn").Pool(processes) as p:
                    for c in range(nmbr_channels):
                        fitpar_event[c] = np.array(
                            p.map(fit_pulse_shape, pulse_event[c, :, :]))

                events.create_dataset('fitpar', data=np.array(fitpar_event))
                # description of the fitparameters (data=column_in_fitpar)
                events['fitpar'].attrs.create(name='t_0', data=0)
                events['fitpar'].attrs.create(name='A_n', data=1)
                events['fitpar'].attrs.create(name='A_t', data=2)
                events['fitpar'].attrs.create(name='tau_n', data=3)
                events['fitpar'].attrs.create(name='tau_in', data=4)
                events['fitpar'].attrs.create(name='tau_t', data=5)

            if calc_sev and calc_nps:

                # ################# STD EVENT #################
                # [pulse_height, t_zero, t_rise, t_max, t_decaystart, t_half, t_end, offset, linear_drift, quadratic_drift]

                sev_pulse_list = []
                sev_fitpar_list = []
                for c in range(nmbr_channels):
                    stdevent_pulse, stdevent_fitpar = generate_standard_event(pulse_event[c, :, :],
                                                                              mainpar_event[c,
                                                                              :, :],
                                                                              pulse_height_interval=[
                                                                                  0.05, 1.5],
                                                                              left_right_cutoff=0.1,
                                                                              rise_time_interval=[
                                                                                  5, 100],
                                                                              decay_time_interval=[
                                                                                  50, 2500],
                                                                              onset_interval=[
                                                                                  1500, 3000],
                                                                              verb=True)
                    sev_pulse_list.append(stdevent_pulse)
                    sev_fitpar_list.append(stdevent_fitpar)

                stdevent = h5f.create_group('stdevent')
                stdevent.create_dataset('event',
                                        data=np.array(sev_pulse_list, dtype=event_dtype))
                stdevent.create_dataset('fitpar',
                                        data=np.array(sev_fitpar_list))
                # description of the fitparameters (data=column_in_fitpar)
                stdevent['fitpar'].attrs.create(name='t_0', data=0)
                stdevent['fitpar'].attrs.create(name='A_n', data=1)
                stdevent['fitpar'].attrs.create(name='A_t', data=2)
                stdevent['fitpar'].attrs.create(name='tau_n', data=3)
                stdevent['fitpar'].attrs.create(name='tau_in', data=4)
                stdevent['fitpar'].attrs.create(name='tau_t', data=5)

                stdevent.create_dataset('mainpar',
                                        data=np.array([calc_main_parameters(x).getArray() for x in sev_pulse_list]))
                # description of the mainpar (data=col_in_mainpar)
                stdevent['mainpar'].attrs.create(name='pulse_height', data=0)
                stdevent['mainpar'].attrs.create(name='t_zero', data=1)
                stdevent['mainpar'].attrs.create(name='t_rise', data=2)
                stdevent['mainpar'].attrs.create(name='t_max', data=3)
                stdevent['mainpar'].attrs.create(name='t_decaystart', data=4)
                stdevent['mainpar'].attrs.create(name='t_half', data=5)
                stdevent['mainpar'].attrs.create(name='t_end', data=6)
                stdevent['mainpar'].attrs.create(name='offset', data=7)
                stdevent['mainpar'].attrs.create(name='linear_drift', data=8)
                stdevent['mainpar'].attrs.create(name='quadratic_drift', data=9)

        # ################# PROCESS NOISE #################
        # if we filtered for noise
        if -1.0 in tpa_list:
            print('WORKING ON EVENTS WITH TPA = -1.')

            metainfo_noise = metainfo[:, metainfo[0, :, 12] == -1.0, :]
            pulse_noise = pulse[:, metainfo[0, :, 12] == -1.0, :]

            print('CREATE DATASET WITH NOISE.')
            noise = h5f.create_group('noise')
            noise.create_dataset('event', data=np.array(pulse_noise, dtype=event_dtype))
            noise.create_dataset('hours', data=np.array(metainfo_noise[0, :, 10]))
            noise.create_dataset('time_s', data=np.array(metainfo_noise[0, :, 4]), dtype='int32')
            noise.create_dataset('time_mus', data=np.array(metainfo_noise[0, :, 5]), dtype='int32')

            if calc_nps:
                if np.shape(pulse_noise)[1] != 0:
                    mean_nps_all = []
                    for c in range(nmbr_channels):
                        mean_nps, _ = calculate_mean_nps(pulse_noise[c, :, :])
                        mean_nps_all.append(mean_nps)
                    frequencies = np.fft.rfftfreq(len(pulse_noise[0, 0]), d=1. / sample_frequency)
                    noise.create_dataset('nps', data=np.array(mean_nps_all))
                    noise.create_dataset('freq', data=frequencies)

                else:
                    print("DataError: No existing noise data for this channel")

        if (-1.0 in tpa_list) and (0 in tpa_list) and calc_sev and calc_nps:
            # ################# OPTIMUMFILTER #################
            # H = optimal_transfer_function(standardevent, mean_nps)
            print('CREATE OPTIMUM FILTER.')

            of = np.array([optimal_transfer_function(sev, nps)
                           for sev, nps in zip(sev_pulse_list, mean_nps_all)])

            optimumfilter = h5f.create_group('optimumfilter')
            optimumfilter.create_dataset('optimumfilter_real',
                                         data=of.real)
            optimumfilter.create_dataset('optimumfilter_imag',
                                         data=of.imag)

        # ################# PROCESS TESTPULSES #################
        # if we filtered for testpulses
        if any(el > 0 for el in tpa_list):
            print('WORKING ON EVENTS WITH TPA > 0.')
            tp_list = np.logical_and(
                metainfo[0, :, 12] != -1.0, metainfo[0, :, 12] != 0.0)

            metainfo_tp = metainfo[:, tp_list, :]
            pulse_tp = pulse[:, tp_list, :]

            nmbr_tp = len(metainfo_tp[0])

            print('CREATE DATASET WITH TESTPULSES.')
            testpulses = h5f.create_group('testpulses')
            testpulses.create_dataset('event', data=np.array(pulse_tp, dtype=event_dtype))
            testpulses.create_dataset(
                'hours', data=np.array(metainfo_tp[0, :, 10]))
            testpulses.create_dataset(
                'testpulseamplitude', data=np.array(metainfo_tp[0, :, 12]))
            testpulses.create_dataset('time_s', data=np.array(metainfo_tp[0, :, 4]), dtype='int32')
            testpulses.create_dataset('time_mus', data=np.array(metainfo_tp[0, :, 5]), dtype='int32')

            if calc_mp:
                print('CALCULATE MP.')

                # basically a for loop running on 4 processes
                mainpar_tp = np.empty([nmbr_channels, nmbr_tp, 10], dtype=float)
                with get_context("spawn").Pool(processes) as p:
                    for c in range(nmbr_channels):
                        mainpar_list_tp = p.map(
                            calc_main_parameters, pulse_tp[c, :, :])
                        mainpar_tp[c] = np.array(
                            [o.getArray() for o in mainpar_list_tp])

                testpulses.create_dataset('mainpar', data=np.array(mainpar_tp))
                # description of the mainpar (data=col_in_mainpar)
                testpulses['mainpar'].attrs.create(name='pulse_height', data=0)
                testpulses['mainpar'].attrs.create(name='t_zero', data=1)
                testpulses['mainpar'].attrs.create(name='t_rise', data=2)
                testpulses['mainpar'].attrs.create(name='t_max', data=3)
                testpulses['mainpar'].attrs.create(name='t_decaystart', data=4)
                testpulses['mainpar'].attrs.create(name='t_half', data=5)
                testpulses['mainpar'].attrs.create(name='t_end', data=6)
                testpulses['mainpar'].attrs.create(name='offset', data=7)
                testpulses['mainpar'].attrs.create(name='linear_drift', data=8)
                testpulses['mainpar'].attrs.create(name='quadratic_drift', data=9)

            if calc_fit:
                print('CALCULATE FIT.')

                fitpar_tp = []

                with get_context("spawn").Pool(processes) as p:
                    for c in range(nmbr_channels):
                        fitpar_tp.append(np.array(
                            p.map(fit_pulse_shape, pulse_tp[c, :, :])))

                testpulses.create_dataset('fitpar', data=np.array(fitpar_tp))
                # description of the fitparameters (data=column_in_fitpar)
                testpulses['fitpar'].attrs.create(name='t_0', data=0)
                testpulses['fitpar'].attrs.create(name='A_n', data=1)
                testpulses['fitpar'].attrs.create(name='A_t', data=2)
                testpulses['fitpar'].attrs.create(name='tau_n', data=3)
                testpulses['fitpar'].attrs.create(name='tau_in', data=4)
                testpulses['fitpar'].attrs.create(name='tau_t', data=5)
