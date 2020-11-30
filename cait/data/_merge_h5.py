
# imports

import h5py
import numpy as np

# function

def merge_h5_sets(path_h5_a, path_h5_b, path_h5_merged,
                  groups_to_merge=['events', 'testpulses', 'noise'],
                  sets_to_merge=['event', 'mainpar', 'true_ph', 'true_onset',
                                 'hours', 'labels', 'testpulseamplitude', 'time_s', 'time_mus'],
                  concatenate_axis=[1, 1, 1, 0, 0, 1, 0, 0, 0],
                  groups_from_a=[],
                  groups_from_b=[],
                  continue_hours = False):
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

    a = h5py.File(path_h5_a, 'r')
    b = h5py.File(path_h5_b, 'r')
    m = h5py.File(path_h5_merged, 'w')

    # merge the groups
    for group in groups_to_merge:

        print('MERGE GROUP: {}.'.format(group))

        if group in a.keys() and group in b.keys():

            # create group in hdf5
            g = m.create_group(group)

            for i, set in enumerate(sets_to_merge):

                print('SET: {}.'.format(set))

                if set in list(a[group].keys()) and set in list(b[group].keys()):

                    print('creating ...')

                    if continue_hours and set == 'hours':
                        last_hour = a[group][set][-1]
                        data = np.concatenate((a[group][set],
                                               b[group][set] + last_hour), axis=concatenate_axis[i])
                    else:
                        data = np.concatenate((a[group][set],
                                               b[group][set]), axis=concatenate_axis[i])

                    # create set in hdf5
                    g.create_dataset(set, data=data)

    # take the groups from a and b
    for f, g_lists in zip([a, b], [groups_from_a, groups_from_b]):
        for group in g_lists:
            if group in list(f.keys()):
                print('COPY GROUP: {}.'.format(group))
                g = m.create_group(group)
                for set in f[group].keys():
                    print('SET: {}.'.format(set))
                    print('creating ...')
                    g.create_dataset(set, data=f[group][set])

    a.close()
    b.close()
    m.close()

    print('Merge done.')