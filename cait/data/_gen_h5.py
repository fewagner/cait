# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os
from multiprocessing import get_context  # Pool
import h5py
import numpy as np
from ..data._raw import read_rdt_file
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
                         processes=4,
                         chunk_size=1000):
    """
    Generates a HDF5 File from an RDT File, optionally MP, Fit, SEV Calculation
    :param path_rdt: string, the full path to the RDT File, e.g. "data/bcks/"
    :param fname: string, the name of the rdt file without appendix, e.g. "bck_001"
    :param path_h5: string, the path where to store the hdf file, e.g. "data/hdf5s/"
    :param channels: list, the numbers of the channels in the hdf5 file that we want to include in rdt
    :param tpa_list: list, the TPAs that we want to include in the hdf5 file, 0 -> events, -1 -> noise, 1 -> TP
    :param calc_mp: bool, calculate and store the main parameters in the hdf5
    :param calc_fit: bool, calculate the parametric pulse fit and store parameters
    :param calc_sev: bool, calculate the standard event and store it in the hdf5
    :param processes: int, the number of processes for the parallel calculation
    :param chunk_size: int, the init size of the arrays, arrays get resized after reaching this size,
        ideally this is just a bit larger than the number of events we want to read from the file
    :return: -
    """

    nmbr_channels = len(channels)

    if not os.path.exists(path_h5):
        os.makedirs(path_h5)

    print('READ EVENTS FROM RDT FILE.')

    metainfo, pulse = \
        read_rdt_file(fname=fname,
                      path=path_rdt,
                      channels=channels,
                      tpa_list=tpa_list,
                      read_events=-1,
                      chunk_size=chunk_size)

    # ipdb.set_trace()

    if nmbr_channels == 2:
        h5f = h5py.File("{}{}-P_Ch{}-L_Ch{}.h5".format(path_h5, fname,
                                                       channels[0], channels[1]), 'w')
    elif nmbr_channels > 2:
        path = "{}{}-P_Ch{}-L_Ch{}".format(path_h5, fname)
        for i, c in enumerate(channels):
            path += '-{}_Ch{}'.format(i+1, c)
        path += ".h5"
        h5f = h5py.File(path, 'w')

    # line below causes an error message (bug: https://github.com/h5py/h5py/issues/1180)
    # h5py.get_config().track_order = True # attrs don't get sorted

    for i, c in enumerate(channels):
        h5f.attrs.create('Ch_{}'.format(i+1), data=c)

    # ################# PROCESS EVENTS #################
    # if we filtered for events
    if 0.0 in tpa_list:

        print('WORKING ON EVENTS WITH TPA = 0.')

        metainfo_event = metainfo[:, metainfo[0, :, 12] == 0, :]
        pulse_event = pulse[:, metainfo[0, :, 12] == 0, :]

        nmbr_events = len(metainfo_event[0])

        events = h5f.create_group('events')
        events.create_dataset('event', data=np.array(pulse_event))
        events.create_dataset('hours', data=np.array(metainfo_event[0, :, 10]))
        print('CREATE DATASET WITH EVENTS.')

        # for small numbers of events only additional overhead is introduced
        processes = 1 if pulse_event.shape[1] < 5 else processes

        if calc_mp:

            print('CALCULATE MAIN PARAMETERS.')

            mainpar_event = np.empty([nmbr_channels, nmbr_events, 10], dtype=float) # 10 is number main parameters

            # basically a for loop running on multiple processes
            with get_context("spawn").Pool(processes) as p:
                for c in range(nmbr_channels):
                    mainpar_list_event = p.map(
                        calc_main_parameters, pulse_event[c, :, :])
                    mainpar_event[c, :, :] = np.array([o.getArray() for o in mainpar_list_event])

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

            fitpar_event = np.empty([nmbr_channels, nmbr_events, 6], dtype=float)  # 6 is number fit parameters

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

        if calc_sev:

            # ################# STD EVENT #################
            # [pulse_height, t_zero, t_rise, t_max, t_decaystart, t_half, t_end, offset, linear_drift, quadratic_drift]


            sev_pulse_list = []
            sev_fitpar_list = []
            for c in range(nmbr_channels):
                stdevent_pulse, stdevent_fitpar = generate_standard_event(pulse_event[c, :, :],
                                                                              mainpar_event[c,
                                                                                            :, :],
                                                                              pulse_height_intervall=[
                                                                                  0.05, 1.5],
                                                                              left_right_cutoff=0.1,
                                                                              rise_time_intervall=[
                                                                                  5, 100],
                                                                              decay_time_intervall=[
                                                                                  50, 2500],
                                                                              onset_intervall=[
                                                                                  1500, 3000],
                                                                              verb=True)
                sev_pulse_list.append(stdevent_pulse)
                sev_fitpar_list.append(stdevent_fitpar)

            stdevent = h5f.create_group('stdevent')
            stdevent.create_dataset('event',
                                    data=np.array(sev_pulse_list))
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

        metainfo_noise = metainfo[:, metainfo[1, :, 12] == -1.0, :]
        pulse_noise = pulse[:, metainfo[1, :, 12] == -1.0, :]

        print('CREATE DATASET WITH NOISE.')
        noise = h5f.create_group('noise')
        noise.create_dataset('event', data=np.array(pulse_noise))
        noise.create_dataset('hours', data=np.array(metainfo_noise[0, :, 10]))

        mean_nps_all = []
        for c in range(nmbr_channels):
            mean_nps, _ = calculate_mean_nps(pulse_noise[c, :, :])
            mean_nps_all.append(mean_nps)
        noise.create_dataset('nps', data=np.array(mean_nps_all))

    if (-1.0 in tpa_list) and (0 in tpa_list) and calc_sev:
        # ################# OPTIMUMFILTER #################
        # H = optimal_transfer_function(standardevent, mean_nps)
        print('CREATE OPTIMUM FILTER.')

        of = np.array([optimal_transfer_function(sev, nps) for sev, nps in zip(sev_pulse_list, mean_nps_all)])

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
            metainfo[1, :, 12] != -1.0, metainfo[1, :, 12] != 0.0)

        metainfo_tp = metainfo[:, tp_list, :]
        pulse_tp = pulse[:, tp_list, :]

        nmbr_tp = len(metainfo_tp[0])

        print('CREATE DATASET WITH TESTPULSES.')
        testpulses = h5f.create_group('testpulses')
        testpulses.create_dataset('event', data=np.array(pulse_tp))
        testpulses.create_dataset(
            'hours', data=np.array(metainfo_tp[0, :, 10]))
        testpulses.create_dataset(
            'testpulseamplitude', data=np.array(metainfo_tp[0, :, 12]))

        if calc_mp:
            print('CALCULATE MP.')

            # basically a for loop running on 4 processes
            mainpar_tp = np.empty([nmbr_channels, nmbr_tp, 10], dtype=float)
            with get_context("spawn").Pool(processes) as p:
                for c in range(nmbr_channels):
                    mainpar_list_tp = p.map(
                        calc_main_parameters, pulse_tp[c, :, :])
                    mainpar_tp[c] = np.array([o.getArray() for o in mainpar_list_tp])

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


    h5f.close()

# ------------------------------------------------------------
# MAIN ROUTINE
# ------------------------------------------------------------


if __name__ == '__main__':
    gen_dataset_from_rdt(path_rdt='./data/run33_TUM40/',
                         fname='bck_056',
                         path_h5='./data/run33_TUM40/',
                         channels=[0, 1],
                         tpa_list=[0.0],
                         calc_fit=False,
                         processes=4)
