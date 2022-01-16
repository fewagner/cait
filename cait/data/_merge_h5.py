# imports

import h5py
import numpy as np
import os
import itertools
import tracemalloc
import time


# function
import h5py

def merge_h5_sets(path_h5_a, path_h5_b, path_h5_merged,
                  groups_to_merge=['events', 'testpulses', 'noise', 'controlpulses', 'stream'],
                  sets_to_merge=['event', 'mainpar', 'true_ph', 'true_onset', 'of_ph', 'sev_fit_par', 'sev_fit_rms',
                                 'hours', 'labels', 'testpulseamplitude', 'time_s', 'time_mus', 'pulse_height',
                                 'pca_error', 'pca_projection', 'tp_hours', 'tp_time_mus', 'tp_time_s', 'tpa',
                                 'trigger_hours', 'trigger_time_mus', 'trigger_time_s'],
                  concatenate_axis=[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  groups_from_a=[],
                  groups_from_b=[],
                  a_name='keep',
                  b_name='keep',
                  continue_hours=True,
                  second_file_start=None,
                  keep_original_files=True,
                  verb=False,
                  trace=False,
                  ):
    """
    Merges two HDF5 files.

    :param path_h5_a: Path to the first file to merge.
    :type path_h5_a: string
    :param path_h5_b: Path to the other file to merge.
    :type path_h5_b: string
    :param path_h5_merged: Path where the merged file is saved.
    :type path_h5_merged: string
    :param groups_to_merge: The groups that hold the arrays that we want to concatenate.
    :type groups_to_merge: list of strings
    :param sets_to_merge: The sets that hold the arrays we want to concatenate, same sets for all groups.
    :type sets_to_merge: list of strings
    :param concatenate_axis: The axis along which the arrays are concatenated. Each n'th index in this list corresponds
        to the n'th string in the sets_to_merge list. If -1, the set is originally a scalar and gets reshaped to a 1D
        array after merge.
    :type concatenate_axis: list of ints
    :param groups_from_a: Which groups are copied from the first HDF5 set.
    :type groups_from_a: list of strings
    :param groups_from_b: Which groups are copied from the second HDF5 set.
    :type groups_from_b: list of strings
    :param a_name: Type a name for the first HDF5 set to identify the data later on with the original data set. This
        name is stored in the origin data set in the corresponding group. If 'keep', the content of the origin data set
        from the HDF5 set is copied.
    :type a_name: string
    :param b_name: Type a name for the second HDF5 set to identify the data later on with the original data set. This
        name is stored in the origin data set in the corresponding group. If 'keep', the content of the origin data set
        from the HDF5 set is copied.
    :type b_name: string
    :param continue_hours: If True, the value of the last hours in a is added to the hours in b.
    :type continue_hours: bool
    :param second_file_start: The hours value at which the second file starts. If this is not handed and continue_hours
        is activated, the value is extracted from the test pulses.
    :type second_file_start: float or None
    :param keep_original_files: If False, the original files are deleted after the merge.
    :type keep_original_files: bool
    :param verb: If True, verbal feedback about the process of the merge is given.
    :type verb: bool
    :param trace: Traces the memory and runtime consumption.
    :type trace: bool
    """

    if trace:
        tracemalloc.start()
        start_time = time.time()

    for v in concatenate_axis:
        if v not in [-1, 0, 1]:
            raise KeyError('The concatenation axis must all be either 0, 1 or -1!')

    with h5py.File(path_h5_a, 'r') as a, h5py.File(path_h5_b, 'r') as b, h5py.File(path_h5_merged, 'w') as m:

        # define hours gap
        if continue_hours:
            if second_file_start is None:
                if 'testpulses' in a and 'testpulses' in b:
                    second_file_start = a['testpulses']['hours'][-1] + (
                            b['testpulses']['time_s'][0] + 10e-6 * b['testpulses']['time_mus'][0] -
                            a['testpulses']['time_s'][-1] - 10e-6 * a['testpulses']['time_mus'][
                                -1]) / 3600
                else:
                    raise KeyError('continue_hours argument requires testpulses group in both files, or needs the start of'
                                   'the second file (second_file_hours) handed seperately!')

        # merge the groups
        for group in groups_to_merge:

            if verb:
                print('--> MERGE GROUP: {}.'.format(group))

            if group in a.keys() and group in b.keys():

                # create group in hdf5
                g = m.create_group(group)

                for i, set in enumerate(sets_to_merge):

                    if verb:
                        print('SET: {}.'.format(set))

                    if set in list(a[group].keys()) and set in list(b[group].keys()):

                        if verb:
                            print('creating ...')

                        # get shape
                        shape_a = a[group][set].shape
                        shape_b = b[group][set].shape
                        shape_m = np.copy(shape_a)
                        if concatenate_axis[i] == -1:
                            if len(shape_a) == 0:
                                shape_m = [2, ]
                            else:
                                shape_m[0] += 1
                        else:
                            shape_m[concatenate_axis[i]] += shape_b[concatenate_axis[i]]

                        # get dtype
                        data_type = a[group][set].dtype

                        # create set in new hdf 5 file
                        g.create_dataset(set,
                                         shape=shape_m,
                                         dtype=data_type)

                        # write data to new file
                        if continue_hours and set in ['hours', 'trigger_hours', 'tp_hours']:
                            g[set][:shape_a[0]] = a[group][set][:]
                            g[set][shape_a[0]:] = b[group][set][:] + second_file_start
                        elif concatenate_axis[i] == 0:
                            g[set][:shape_a[0]] = a[group][set][:]
                            g[set][shape_a[0]:] = b[group][set][:]
                        elif concatenate_axis[i] == 1:
                            for c in range(shape_a[0]):
                                g[set][c, :shape_a[1]] = a[group][set][c, :]
                                g[set][c, shape_a[1]:] = b[group][set][c, :]
                        elif concatenate_axis[i] == -1:
                            if len(shape_a) == 0:
                                g[set][0] = a[group][set][()]
                            else:
                                g[set][:-1] = a[group][set][:]
                            g[set][-1] = b[group][set][()]

                        # add the labels list
                        if set == 'labels':
                            g[set].attrs.create(name='unlabeled', data=0)
                            g[set].attrs.create(name='Event_Pulse', data=1)
                            g[set].attrs.create(name='Test/Control_Pulse', data=2)
                            g[set].attrs.create(name='Noise', data=3)
                            g[set].attrs.create(name='Squid_Jump', data=4)
                            g[set].attrs.create(name='Spike', data=5)
                            g[set].attrs.create(name='Early_or_late_Trigger', data=6)
                            g[set].attrs.create(name='Pile_Up', data=7)
                            g[set].attrs.create(name='Carrier_Event', data=8)
                            g[set].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
                            g[set].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
                            g[set].attrs.create(name='Decaying_Baseline', data=11)
                            g[set].attrs.create(name='Temperature_Rise', data=12)
                            g[set].attrs.create(name='Stick_Event', data=13)
                            g[set].attrs.create(name='Square_Waves', data=14)
                            g[set].attrs.create(name='Human_Disturbance', data=15)
                            g[set].attrs.create(name='Large_Sawtooth', data=16)
                            g[set].attrs.create(name='Cosinus_Tail', data=17)
                            g[set].attrs.create(name='Light_only_Event', data=18)
                            g[set].attrs.create(name='Ring_Light_Event', data=19)
                            g[set].attrs.create(name='Sharp_Light_Event', data=20)
                            g[set].attrs.create(name='unknown/other', data=99)

                        # add the mainpar labels
                        if len(set) >= 7:
                            if set[:7] == 'mainpar':
                                g[set].attrs.create(name='pulse_height', data=0)
                                g[set].attrs.create(name='t_zero', data=1)
                                g[set].attrs.create(name='t_rise', data=2)
                                g[set].attrs.create(name='t_max', data=3)
                                g[set].attrs.create(name='t_decaystart', data=4)
                                g[set].attrs.create(name='t_half', data=5)
                                g[set].attrs.create(name='t_end', data=6)
                                g[set].attrs.create(name='offset', data=7)
                                g[set].attrs.create(name='linear_drift', data=8)
                                g[set].attrs.create(name='quadratic_drift', data=9)

                         # add the additional mainpar labels
                        if len(set) >= 11:
                            if set[:11] == 'add_mainpar':
                                g[set].attrs.create(name='array_max', data=0)
                                g[set].attrs.create(name='array_min', data=1)
                                g[set].attrs.create(name='var_first_eight', data=2)
                                g[set].attrs.create(name='mean_first_eight', data=3)
                                g[set].attrs.create(name='var_last_eight', data=4)
                                g[set].attrs.create(name='mean_last_eight', data=5)
                                g[set].attrs.create(name='var', data=6)
                                g[set].attrs.create(name='mean', data=7)
                                g[set].attrs.create(name='skewness', data=8)
                                g[set].attrs.create(name='max_derivative', data=9)
                                g[set].attrs.create(name='ind_max_derivative', data=10)
                                g[set].attrs.create(name='min_derivative', data=11)
                                g[set].attrs.create(name='ind_min_derivative', data=12)
                                g[set].attrs.create(name='max_filtered', data=13)
                                g[set].attrs.create(name='ind_max_filtered', data=14)
                                g[set].attrs.create(name='skewness_filtered_peak', data=15)

                            if set[:11] == 'sev_fit_par':
                                g[set].attrs.create(name='pulse_height', data=0)
                                g[set].attrs.create(name='onset', data=1)
                                g[set].attrs.create(name='constant_coefficient', data=2)
                                g[set].attrs.create(name='linear_coefficient', data=3)
                                g[set].attrs.create(name='quadratic_coefficient', data=4)
                                g[set].attrs.create(name='cubic_coefficient', data=5)

                if 'event' in a[group] and 'event' in b[group]:

                    nmbr_a = len(a[group]['event'][0])
                    nmbr_b = len(b[group]['event'][0])
                    nmbr_m = nmbr_a + nmbr_b

                    # write the original file names to dataset
                    string_dt = h5py.special_dtype(vlen=str)
                    orig = g.require_dataset('origin',
                                         shape=(nmbr_m,),
                                         dtype=string_dt)

                    if a_name == 'keep':
                        if 'origin' in a[group]:
                            orig[:nmbr_a] = a[group]['origin']
                        else:
                            orig[:nmbr_a] = 'a'
                    else:
                        orig[:nmbr_a] = a_name

                    if b_name == 'keep':
                        if 'origin' in b[group]:
                            orig[nmbr_a:] = b[group]['origin']
                        else:
                            orig[nmbr_a:] = 'b'
                    else:
                        orig[nmbr_a:] = b_name

        # take the groups from a and b
        for f, g_lists in zip([a, b], [groups_from_a, groups_from_b]):
            for group in g_lists:
                if group in list(f.keys()):
                    if verb:
                        print('COPY GROUP: {}.'.format(group))
                    g = m.create_group(group)
                    for set in f[group].keys():
                        if verb:
                            print('SET: {}.'.format(set))
                            print('creating ...')
                        g.create_dataset(set, data=f[group][set])

    if not keep_original_files:
        for p in [path_h5_a, path_h5_b]:
            if verb:
                print('Deleting {}.'.format(p))
            os.remove(p)

    print('Merge done.')

    if trace:
        current, peak = tracemalloc.get_traced_memory()
        print(
            f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Runtime was {time.time() - start_time};")
        tracemalloc.stop()