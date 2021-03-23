# imports

import h5py
import numpy as np
import os
import itertools


# function

def merge_h5_sets(path_h5_a, path_h5_b, path_h5_merged,
                  groups_to_merge=['events', 'testpulses', 'noise', 'controlpulses'],
                  sets_to_merge=['event', 'mainpar', 'true_ph', 'true_onset', 'of_ph', 'sev_fit_par', 'sev_fit_rms',
                                 'hours', 'labels', 'testpulseamplitude', 'time_s', 'time_mus', 'pulse_height',
                                 'pca_error', 'pca_projection'],
                  concatenate_axis=[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
                  groups_from_a=[],
                  groups_from_b=[],
                  a_name='keep',
                  b_name='keep',
                  continue_hours=True,
                  keep_original_files=True,
                  verb=False,
                  ):
    """
    Merges two HDF5 files, groups to merge can be chosen

    :param path_h5_a: path to the first file to merge
    :type path_h5_a: string
    :param path_h5_b: path to the other file to merge
    :type path_h5_b: string
    :param path_h5_merged: path where the merged file is saved
    :type path_h5_merged: string
    :param groups_to_merge: the groups that hold the arrays that we want to concatenate
    :type groups_to_merge: list of strings
    :param sets_to_merge: the sets that hold the arrays we want to concatenate, same sets for all groups
    :type sets_to_merge: list of strings
    :param concatenate_axis:
    :type concatenate_axis: list of ints
    :param groups_from_a:
    :type groups_from_a: list of strings
    :param groups_from_b:
    :type groups_from_b: list of strings
    :param continue_hours: bool, if True, the value of the last hours in a is added to the hours in b
    :return: -
    :rtype: -
    """

    for v in concatenate_axis:
        if v not in [0, 1]:
            raise KeyError('The concatenation axis must all be either 0 or 1!')

    with h5py.File(path_h5_a, 'r') as a, h5py.File(path_h5_b, 'r') as b, h5py.File(path_h5_merged, 'w') as m:

        # extract the nmbr of events from both files
        for g, (s, c) in itertools.product(groups_to_merge, zip(sets_to_merge, concatenate_axis)):
            try:
                if c == 0:
                    nmbr_merged = len(a[g][s]) + len(b[g][s])
                    nmbr_a = len(a[g][s])
                elif c == 1:
                    nmbr_merged = len(a[g][s][0]) + len(b[g][s][0])
                    nmbr_a = len(a[g][s][0])
                else:
                    raise IndexError
                if nmbr_merged == 0 or nmbr_a == 0:
                    raise IndexError
                break
            except IndexError:
                pass

        # define hours gap
        if continue_hours:
            if 'testpulses' in a and 'testpulses' in b:
                second_file_hours = a['testpulses']['hours'][-1] + (
                        b['testpulses']['time_s'][0] + 10e-6 * b['testpulses']['time_mus'][0] -
                        a['testpulses']['time_s'][-1] - 10e-6 * a['testpulses']['time_mus'][
                            -1]) / 3600
            else:
                raise KeyError('continue_hours argument requires testpulses group in both files!')

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
                        shape_b = a[group][set].shape
                        shape_m = np.copy(shape_a)
                        shape_m[concatenate_axis[i]] += shape_b[concatenate_axis[i]]

                        # get dtype
                        data_type = a[group][set].dtype

                        # create set in new hdf 5 file
                        g.create_dataset(set,
                                         shape=shape_m,
                                         dtype=data_type)

                        # write data to new file
                        if continue_hours and set == 'hours':
                            g[set][:shape_a[0]] = a[group][set][:]
                            g[set][shape_a[0]:] = b[group][set][:] + second_file_hours
                        elif concatenate_axis[i] == 0:
                            g[set][:shape_a[0]] = a[group][set][:]
                            g[set][shape_a[0]:] = b[group][set][:]
                        elif concatenate_axis[i] == 1:
                            for c in range(shape_a[0]):
                                g[set][c, :shape_a[1]] = a[group][set][c, :]
                                g[set][c, shape_a[1]:] = b[group][set][c, :]


                        # if continue_hours and set == 'hours':
                        #     data = np.concatenate((a[group][set],
                        #                            b[group][set] + second_file_hours), axis=concatenate_axis[i])
                        # else:
                        #     data = np.concatenate((a[group][set],
                        #                            b[group][set]), axis=concatenate_axis[i])

                        # create set in hdf5
                        #g.create_dataset(set, data=data)

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

        # write the original file names to group
        orig = m.require_group('origin')
        string_dt = h5py.special_dtype(vlen=str)
        orig.require_dataset('filename',
                             shape=(nmbr_merged,),
                             dtype=string_dt)

        if a_name == 'keep':
            if 'origin' in a:
                orig['filename'][:nmbr_a] = a['origin']['filename']
            else:
                orig['filename'][:nmbr_a] = 'a'
        else:
            orig['filename'][:nmbr_a] = a_name

        if b_name == 'keep':
            if 'origin' in b:
                orig['filename'][nmbr_a:] = b['origin']['filename']
            else:
                orig['filename'][nmbr_a:] = 'b'
        else:
            orig['filename'][nmbr_a:] = b_name

    if not keep_original_files:
        for p in [path_h5_a, path_h5_b]:
            if verb:
                print('Deleting {}.'.format(p))
            os.remove(p)

    print('Merge done.')
