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

# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------


def gen_dataset_from_rdt(path_rdt,
                         fname,
                         path_h5,
                         phonon_channel,
                         light_channel,
                         tpa_list=[0.0],
                         calc_mp=True,
                         calc_fit=False,
                         calc_sev=False,
                         processes=4):

    if not os.path.exists(path_h5):
        os.makedirs(path_h5)

    print('READ EVENTS FROM RDT FILE.')

    metainfo, pulse = \
        read_rdt_file(fname=fname,
                      path=path_rdt,
                      phonon_channel=phonon_channel,
                      light_channel=light_channel,
                      tpa_list=tpa_list,
                      read_events=-1)

    h5f = h5py.File("{}{}-P_Ch{}-L_Ch{}.h5".format(path_h5, fname,
                                                   phonon_channel, light_channel), 'w')

    # line below causes an error message (bug: https://github.com/h5py/h5py/issues/1180)
    # h5py.get_config().track_order = True # attrs don't get sorted

    h5f.attrs.create('phonon_channel', data=phonon_channel)
    h5f.attrs.create('light_channel', data=light_channel)

    # ################# PROCESS EVENTS #################
    # if we filtered for events
    if 0.0 in tpa_list:

        print('WORKING ON EVENTS WITH TPA = 0.')

        metainfo_event = metainfo[:, metainfo[1, :, 12] == 0, :]
        pulse_event = pulse[:, metainfo[1, :, 12] == 0, :]

        events = h5f.create_group('events')
        events.create_dataset('event', data=np.array(pulse_event))
        events.create_dataset('hours', data=np.array(metainfo_event[0, :, 10]))
        print('CREATE DATASET WITH EVENTS.')

        # for small numbers of events only additional overhead is introduced
        processes = 1 if pulse_event.shape[1] < 5 else processes

        if calc_mp:

            print('CALCULATE MAIN PARAMETERS.')

            # basically a for loop running on 4 processes
            with get_context("spawn").Pool(processes) as p:
                p_mainpar_list_event = p.map(
                    calc_main_parameters, pulse_event[0, :, :])
                l_mainpar_list_event = p.map(
                    calc_main_parameters, pulse_event[1, :, :])
            mainpar_event = np.array([[o.getArray() for o in p_mainpar_list_event],
                                      [o.getArray() for o in l_mainpar_list_event]])

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

            with get_context("spawn").Pool(processes) as p:
                p_fitpar_event = np.array(
                    p.map(fit_pulse_shape, pulse_event[0, :, :]))
                l_fitpar_event = np.array(
                    p.map(fit_pulse_shape, pulse_event[1, :, :]))

            fitpar_event = np.array([p_fitpar_event, l_fitpar_event])

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
            p_stdevent_pulse, p_stdevent_fitpar = generate_standard_event(pulse_event[0, :, :],
                                                                          mainpar_event[0,
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
            l_stdevent_pulse, l_stdevent_fitpar = generate_standard_event(pulse_event[1, :, :],
                                                                          mainpar_event[1,
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

            stdevent = h5f.create_group('stdevent')
            stdevent.create_dataset('event',
                                    data=np.array([p_stdevent_pulse, l_stdevent_pulse]))
            stdevent.create_dataset('fitpar',
                                    data=np.array([p_stdevent_fitpar, l_stdevent_fitpar]))
            # description of the fitparameters (data=column_in_fitpar)
            stdevent['fitpar'].attrs.create(name='t_0', data=0)
            stdevent['fitpar'].attrs.create(name='A_n', data=1)
            stdevent['fitpar'].attrs.create(name='A_t', data=2)
            stdevent['fitpar'].attrs.create(name='tau_n', data=3)
            stdevent['fitpar'].attrs.create(name='tau_in', data=4)
            stdevent['fitpar'].attrs.create(name='tau_t', data=5)

            stdevent.create_dataset('mainpar',
                                    data=np.array([calc_main_parameters(p_stdevent_pulse).getArray(),
                                                   calc_main_parameters(l_stdevent_pulse).getArray()]))
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

        p_mean_nps, _ = calculate_mean_nps(
            pulse_noise[0, :, :], record_length=len(pulse_noise[0, 0]))
        l_mean_nps, _ = calculate_mean_nps(
            pulse_noise[1, :, :], record_length=len(pulse_noise[0, 0]))
        noise.create_dataset('nps', data=np.array([p_mean_nps, l_mean_nps]))

    if (-1.0 in tpa_list) and (0 in tpa_list) and calc_sev:
        # ################# OPTIMUMFILTER #################
        # H = optimal_transfer_function(standardevent, mean_nps)
        print('CREATE OPTIMUM FILTER.')
        optimumfilter = h5f.create_group('optimumfilter')
        optimumfilter.create_dataset('optimumfilter',
                                     data=np.array([optimal_transfer_function(p_stdevent_pulse, p_mean_nps),
                                                    optimal_transfer_function(l_stdevent_pulse, l_mean_nps)]))

    # ################# PROCESS TESTPULSES #################
    # if we filtered for testpulses
    if any(el > 0 for el in tpa_list):
        print('WORKING ON EVENTS WITH TPA > 0.')
        tp_list = np.logical_and(
            metainfo[1, :, 12] != -1.0, metainfo[1, :, 12] != 0.0)

        metainfo_tp = metainfo[:, tp_list, :]
        pulse_tp = pulse[:, tp_list, :]

        print('CREATE DATASET WITH TESTPULSES.')
        testpulses = h5f.create_group('testpulses')
        testpulses.create_dataset('event', data=np.array(pulse_tp))
        testpulses.create_dataset(
            'hours', data=np.array(metainfo_tp[0, :, 10]))
        testpulses.create_dataset(
            'testpulseamplitute', data=np.array(metainfo_tp[0, :, 12]))

        if calc_mp:
            print('CALCULATE MP.')

            # basically a for loop running on 4 processes
            with get_context("spawn").Pool(processes) as p:
                p_mainpar_list_tp = p.map(
                    calc_main_parameters, pulse_tp[0, :, :])
            p_mainpar_tp = np.array([o.getArray() for o in p_mainpar_list_tp])

            with get_context("spawn").Pool(processes) as p:
                l_mainpar_list_tp = p.map(
                    calc_main_parameters, pulse_tp[1, :, :])
            l_mainpar_tp = np.array([o.getArray() for o in l_mainpar_list_tp])

            mainpar_tp = np.array([p_mainpar_tp, l_mainpar_tp])

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

            with get_context("spawn").Pool(processes) as p:
                p_fitpar_tp = np.array(
                    p.map(fit_pulse_shape, pulse_tp[0, :, :]))
                l_fitpar_tp = np.array(
                    p.map(fit_pulse_shape, pulse_tp[1, :, :]))

            fitpar_tp = np.array([p_fitpar_tp, l_fitpar_tp])

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
                         phonon_channel=6,
                         light_channel=7,
                         tpa_list=[0.0],
                         calc_fit=False,
                         processes=4)
