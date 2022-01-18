# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os
import h5py
import numpy as np
from ..data._raw import convert_to_V
from tqdm.auto import tqdm, trange
import time
import tracemalloc


# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------


def gen_dataset_from_rdt_memsafe(path_rdt,
                                 fname,
                                 path_h5,
                                 channels,
                                 tpa_list=[0., 1., -1.],
                                 event_dtype='float32',
                                 ints_in_header=7,
                                 dvm_channels=0,
                                 record_length=16384,
                                 batch_size=1000,
                                 trace=False,
                                 indiv_tpas=False,
                                 ):
    """
    Generates a HDF5 File from an RDT File, with an memory safe implementation. This is recommended, in case the RDT
    file is large or the available RAM small.

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
    :param event_dtype: Datatype to save the events with.
    :type event_dtype: string
    :param ints_in_header: The number of ints in the header of the events in the RDF file. This should be either
            7 or 6!
    :type ints_in_header: int
    :param dvm_channels: The number of DVM channels, this can be read in the PAR file.
    :type dvm_channels: int
    :param record_length: The number of samples in one record window.
    :type record_length: int
    :param batch_size: The batch size for loading the samples from disk. Usually 1000 is a good value and produces
        RAM usage around 250 MB.
    :type batch_size: int
    :param trace: Trace the runtime and memory consumption
    :type trace: bool
    :param individual_tpas: Write individual TPAs for the all channels. This results in a testpulseamplitude dataset
            of shape (nmbr_channels, nmbr_testpulses). Otherwise we have (nmbr_testpulses).
    :type individual_tpas: bool
    """

    nmbr_channels = len(channels)

    if not os.path.exists(path_h5):
        os.makedirs(path_h5)

    print('\nREAD EVENTS FROM RDT FILE.')

    if trace:
        tracemalloc.start()
        start_time = time.time()

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

    # header = np.dtype([('detector_nmbr', 'i4'),
    #                    ('coincide_pulses', 'i4'),
    #                    ('trig_count', 'i4'),
    #                    ('trig_delay', 'i4'),
    #                    ('abs_time_s', 'i4'),
    #                    ('abs_time_mus', 'i4'),
    #                    ('delay_ch_tp', 'i4', (int(ints_in_header == 7),)),
    #                    ('time_low', 'i4'),
    #                    ('time_high', 'i4'),
    #                    ('qcd_events', 'i4'),
    #                    ('hours', 'f4'),
    #                    ('dead_time', 'f4'),
    #                    ('test_pulse_amplitude', 'f4'),
    #                    ('dac_output', 'f4'),
    #                    ])

    recs = np.memmap("{}{}.rdt".format(path_rdt, fname), dtype=record, mode='r')

    if trace:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Runtime was {time.time() - start_time};")
        tracemalloc.stop()
        tracemalloc.start()
        start_time = time.time()

    nmbr_all_events = recs.shape[0]

    print('Total Records in File: ', nmbr_all_events)

    # get only consecutive events from these two channels

    if trace:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Runtime was {time.time() - start_time};")
        tracemalloc.stop()
        tracemalloc.start()
        start_time = time.time()

    print('Getting good idx. (Depending on OS and drive reading speed, this might take some minutes!)')

    good_idx = []

    detnmbrs = []

    for r in tqdm(recs):
        detnmbrs.append(r['detector_nmbr'])

    # detnmbrs = np.array(detnmbrs)

    detnmbrs = np.array(recs['detector_nmbr'])

    for c in channels:
        print(f'Event Counts Channel {c}: {np.sum(detnmbrs == c)}')
    recs = np.memmap("{}{}.rdt".format(path_rdt, fname), dtype=record, mode='r')

    for idx in range(recs.shape[0] - nmbr_channels + 1):
        cond = True
        for j in range(nmbr_channels):
            cond = np.logical_and(cond, detnmbrs[idx + j] == channels[j])
            if not cond:
                break
        if cond:
            good_idx.append(idx)

    if trace:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Runtime was {time.time() - start_time};")
        tracemalloc.stop()
        tracemalloc.start()
        start_time = time.time()

    print('Getting good tpas.')

    good_idx = np.array(good_idx)
    good_tpas = np.array(recs[good_idx]['test_pulse_amplitude'])

    if trace:
        current, peak = tracemalloc.get_traced_memory()
        print(
            f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Runtime was {time.time() - start_time};")
        tracemalloc.stop()
        tracemalloc.start()
        start_time = time.time()

    print('Good consecutive counts: {}'.format(good_idx.shape[0]))
    # ----

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
            print('\nWORKING ON EVENTS WITH TPA = 0.')

            nmbr_events = np.sum(good_tpas == 0.)
            idx_events = good_idx[good_tpas == 0.]

            print('CREATE DATASET WITH EVENTS.')
            events = h5f.create_group('events')
            events.create_dataset('event', shape=(nmbr_channels, nmbr_events, record_length), dtype=event_dtype)
            events.create_dataset('hours', data=recs['hours'][idx_events], dtype=float)
            events.create_dataset('dac_output', data=recs['dac_output'][idx_events], dtype=float)
            events.create_dataset('time_s', data=recs['abs_time_s'][idx_events], dtype='int32')
            events.create_dataset('time_mus', data=recs['abs_time_mus'][idx_events], dtype='int32')

            with tqdm(total=nmbr_channels * nmbr_events) as pbar:
                holder = np.zeros((batch_size, record_length), dtype=event_dtype)
                for c in range(nmbr_channels):
                    # create new memmep for lower memory usage
                    del recs
                    recs = np.memmap("{}{}.rdt".format(path_rdt, fname), dtype=record, mode='r')
                    counter = 0
                    while counter < nmbr_events - batch_size:
                        holder[:, :] = convert_to_V(recs['samples'][idx_events[counter:counter + batch_size] + c])
                        events['event'][c, counter:counter + batch_size, ...] = holder
                        pbar.update(batch_size)
                        counter += batch_size
                    events['event'][c, counter:nmbr_events, ...] = convert_to_V(
                        recs['samples'][idx_events[counter:nmbr_events] + c])
                    pbar.update(nmbr_events - counter)

        # ################# PROCESS NOISE #################
        # if we filtered for noise
        if -1.0 in tpa_list:
            print('\nWORKING ON EVENTS WITH TPA = -1.')

            nmbr_noise = np.sum(good_tpas == -1.)
            idx_noise = good_idx[good_tpas == -1.]

            print('CREATE DATASET WITH NOISE.')
            noise = h5f.create_group('noise')
            noise.create_dataset('event', shape=(nmbr_channels, nmbr_noise, record_length), dtype=event_dtype)
            noise.create_dataset('hours', data=recs['hours'][idx_noise], dtype=float)
            noise.create_dataset('dac_output', data=recs['dac_output'][idx_noise], dtype=float)
            noise.create_dataset('time_s', data=recs['abs_time_s'][idx_noise], dtype='int32')
            noise.create_dataset('time_mus', data=recs['abs_time_mus'][idx_noise], dtype='int32')

            with tqdm(total=nmbr_channels * nmbr_noise) as pbar:
                holder = np.zeros((batch_size, record_length), dtype=event_dtype)
                for c in range(nmbr_channels):
                    # create new memmep for lower memory usage
                    del recs
                    recs = np.memmap("{}{}.rdt".format(path_rdt, fname), dtype=record, mode='r')
                    counter = 0
                    while counter < nmbr_noise - batch_size:
                        holder[:, :] = convert_to_V(recs['samples'][idx_noise[counter:counter + batch_size] + c])
                        noise['event'][c, counter:counter + batch_size, ...] = holder
                        pbar.update(batch_size)
                        counter += batch_size
                    noise['event'][c, counter:nmbr_noise, ...] = convert_to_V(
                        recs['samples'][idx_noise[counter:nmbr_noise] + c])
                    pbar.update(nmbr_noise - counter)

        # ################# PROCESS TESTPULSES #################
        # if we filtered for testpulses
        if any(el > 0 for el in tpa_list):
            print('\nWORKING ON EVENTS WITH TPA > 0.')

            nmbr_testpulses = np.sum(good_tpas > 0.)
            idx_testpulses = good_idx[good_tpas > 0.]

            print('CREATE DATASET WITH TESTPULSES.')
            testpulses = h5f.create_group('testpulses')
            testpulses.create_dataset('event', shape=(nmbr_channels, nmbr_testpulses, record_length), dtype=event_dtype)
            data_to_write = recs['test_pulse_amplitude'][idx_testpulses]
            if indiv_tpas:
                data_to_write = np.tile(data_to_write, (nmbr_channels, 1))
            testpulses.create_dataset('testpulseamplitude', data=data_to_write,
                                      dtype=float)
            testpulses.create_dataset('hours', data=recs['hours'][idx_testpulses], dtype=float)
            testpulses.create_dataset('dac_output', data=recs['dac_output'][idx_testpulses], dtype=float)
            testpulses.create_dataset('time_s', data=recs['abs_time_s'][idx_testpulses], dtype='int32')
            testpulses.create_dataset('time_mus', data=recs['abs_time_mus'][idx_testpulses], dtype='int32')

            with tqdm(total=nmbr_channels * nmbr_testpulses) as pbar:
                holder = np.zeros((batch_size, record_length), dtype=event_dtype)
                for c in range(nmbr_channels):
                    # create new memmep for lower memory usage
                    # del recs
                    # recs = np.memmap("{}{}.rdt".format(path_rdt, fname), dtype=record, mode='r')
                    counter = 0
                    while counter < nmbr_testpulses - batch_size:
                        holder[:, :] = convert_to_V(recs['samples'][idx_testpulses[counter:counter + batch_size] + c])
                        testpulses['event'][c, counter:counter + batch_size, ...] = holder
                        pbar.update(batch_size)
                        counter += batch_size
                    testpulses['event'][c, counter:nmbr_testpulses, ...] = convert_to_V(
                        recs['samples'][idx_testpulses[counter:nmbr_testpulses] + c])
                    pbar.update(nmbr_testpulses - counter)

    # recs
    del holder

    if trace:
        current, peak = tracemalloc.get_traced_memory()
        print(
            f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Runtime was {time.time() - start_time};")
        tracemalloc.stop()
