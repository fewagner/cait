import os
from typing import List
import time
import tracemalloc
from deprecation import deprecated

import numpy as np
import h5py
import cait as ai

from ..styles._print_styles import sizeof_fmt, fmt_ds

def ds_source_available(file: h5py.File, group: str, dataset: str):
    """
    Checks whether the sources of a virtual dataset 'dataset' in the group 'group' are still available to the file 'file'. For regular datasets (not virtual) this method returns True.
    This method DOES NOT check for the existence of the dataset!

    :param file: Open file stream to the HDF5 file.
    :type file: h5py.File
    :param group: The HDF5 group we want to check.
    :type group: str
    :param group: The HDF5 dataset in group 'group' we want to check.
    :type group: str

    :return: False if origins of virtual dataset are unavailable. True if available or dataset is regular dataset.
    :rtype: bool
    """
    # check if sources are still available in case a virtual dataset is requested 
    if file[group][dataset].is_virtual:
        sources = file[group][dataset].virtual_sources()
        filenames = [x[1] for x in sources]
        return all([os.path.exists(x) for x in filenames])
    else:
        return True
        
def get_dataset_properties(files: List[str], group: str, dataset: str, src_dir: str = ''):
    """
    Convenience function to get the number of events for a list of files, the dimension along which these events are oriented, the remaining shape of the dataset as well as the dtype. 

    Assesses information from first file (that the files are consistent has to be checked by func:`check_file_consistency`

    :param files: Names of the HDF5 files (without path and .h5 extension).
    :type files: List[str]
    :param group: The HDF5 group of the dataset we want to get the properties from,
    :type group: str
    :param dataset: The HDF5 dataset we want to get the properties from,
    :type dataset: str
    :param src_dir: Source path of the files in the [files] list.
    :type src_dir: str
    
    The following returns a list `[n_events_file1, n_events_file2]` of the number of events for each file, the `shape` of the dataset `event` in group `events` (where the dimension of the events has been dropped, i.e. `len(shape) = dataset.ndim - 1`), the `dtype` of the dataset `event` and the dimension along which the events extend, `events_dim`. See func:`combine` for more information.

    .. code-block:: python

        n_events, shape, dtype, events_dim = get_dataset_properties([file1, file2], "events", "event", "directory")
    """

    with h5py.File(os.path.join(src_dir, files[0] + ".h5"), 'r') as h5f:
        dtype = h5f[group][dataset].dtype
        shape = list(h5f[group][dataset].shape)

        if h5f[group][dataset].ndim == 1: 
            shape.pop(0)
            events_dim = 0
        else: 
            shape.pop(1)
            events_dim = 1

    n_events = list()
    for f in files:
        with h5py.File(os.path.join(src_dir, f + ".h5"), 'r') as h5f:
            n_events.append(h5f[group][dataset].shape[events_dim])

    return n_events, shape, dtype, events_dim

def check_file_consistency(files: List[str], src_dir: str, groups_combine: List[str], groups_include: List[str]):
    """
    Checks whether the groups/datasets in a list of files are ready to be combined/merged. The function checks if all groups in groups_combine are present in all files, and if all the datasets within these groups have appropriate shape (all but the events-dimension have to agree; the events-dimension is 0 for 1-dimensional data and 1 for 2- and 3-dimensional data) and dtype. Finally, it is checked whether the names in groups_include are present in at least one file.

    :param files: List of HDF5 files to be combined (without the .h5 extension)
    :type files: List[str]
    :param src_dir: The directory of the HDF5 files you wish to combine. Default: current directory
    :type src_dir: str
    :param groups_combine: Groups in the HDF5 files you wish to combine. The function will loop through all the datasets within that group and combine them along the first dimension (for 1-dimensional data), or along the second dimension (for 2- and 3-dimensional data). This is because the datasets have shape (events, data) or (channels, events, data) and we want to append them along the events-dimension.
    :type groups_combine: List[str]
    :param groups_include: Groups you just wish to copy from one representative file, i.e. the data will not be appended. This can be useful for SEVs or optimum filter, etc. 
    :type groups_include: List[str]
    """

    # check if all groups_combine exist in all files, that all the datasets are the same and that the non-event-dimensions agree
    for group in groups_combine:
        # first check for existence of datasets
        datasets = []
        for f in files:
            with h5py.File(os.path.join(src_dir, f + ".h5"), 'r') as h5f:
                assert group in list(h5f.keys()), f"Group '{group}' is not present in file '{f}'."
                datasets.append(set(h5f[group].keys()))

        assert all(datasets[0] == s for s in datasets[1:]), f"Datasets of group '{group}' are not consistent across all files."

        # if passed, check for same shape (datasets[0] is representative, as we already made sure that they are all the same)
        for ds in datasets[0]:
            shapes = []
            dtypes = []
            for f in files:
                with h5py.File(os.path.join(src_dir, f + ".h5"), 'r') as h5f:
                    shape, dtype = list(h5f[group][ds].shape), h5f[group][ds].dtype
                    if h5f[group][ds].ndim == 1: shape.pop(0)
                    else: shape.pop(1)
                    
                    shapes.append(shape)
                    dtypes.append(dtype)
            
            assert all(shapes[0] == s for s in shapes[1:]), f"Shapes of dataset '{ds}' in group '{group}' are not consistent across all files."
            assert all(dtypes[0] == d for d in dtypes[1:]), f"dtypes of dataset '{ds}' in group '{group}' are not consistent across all files."

    # check if groups_include are present in at least one file
    all_groups = set()
    for f in files:
            with h5py.File(os.path.join(src_dir, f + ".h5"), 'r') as h5f:
                all_groups = all_groups.union(list(h5f.keys()))

    assert all(g in all_groups for g in groups_include), f"Some groups in 'groups_include' could not be found in either of the files to be combined/merged."
        
def combine_h5(fname: str,
               files: List[str],
               src_dir: str = '',
               out_dir: str = '',
               groups_combine: List[str] = ["events", "testpulses", "noise"],
               groups_include: List[str] = [],
               extend_hours: bool = True
               ):
    """
    Combines multiple HDF5 files into a single file using virtual datasets, i.e. none of the data is actually copied, yet it can be accessed as if it was stored in the same file. It is important that the initial HDF5 files have the same structure, at least for the data groups handed by groups_merge. Otherwise the function might crash or, even worse, yield nonsensical data combinations.

    Be aware, that the files are combined in the order as they are specified in 'files'. If you want to make sure that they are in (temporally) increasing order, you have to sort the list accordingly.

    :param fname: The name of the output file (without the .h5 extension). If it already exists, the file content is overwritten.
    :type fname: str
    :param files: List of HDF5 files to be combined (without the .h5 extension)
    :type files: List[str]
    :param src_dir: The directory of the HDF5 files you wish to combine. Default: current directory
    :type src_dir: str
    :param out_dir: The directory where the output HDF5 file will be saved. Default: current directory
    :type out_dir: str
    :param groups_combine: Groups in the HDF5 files you wish to combine. The function will loop through all the datasets within that group and combine them along the first dimension (for 1-dimensional data), or along the second dimension (for 2- and 3-dimensional data). This is because the datasets have shape (events, data) or (channels, events, data) and we want to append them along the events-dimension.
    :type groups_combine: List[str]
    :param groups_include: Groups you just wish to copy from one representative file, i.e. the data will not be appended. This can be useful for SEVs or optimum filter, etc. 
    :type groups_include: List[str]
    :param extend_hours: If True, the ``hours`` dataset of all groups is updated in the final file such that it does not restart at 0 after every file but continuously increases. This requires the existence of datasets ``event``, ``time_s``, and ``time_mus`` in the respective groups.
    :type extend_hours: bool
    """

    out_path = os.path.join(out_dir, fname + ".h5")

    # checks before start of combination
    if os.path.exists(out_path):
        print(f"Overwriting existing file '{out_path}'.")
        os.remove(out_path)

    assert len(files) > 1, "At least two files have to be chosen to merge."

    check_file_consistency(files, src_dir, groups_combine, groups_include)

    # Include groups from groups_combine (i.e. the ones that are appended along the events-dimension)
    with h5py.File(out_path, 'a') as out_file:
        for g in groups_combine:
            current_group = out_file.require_group(g)

            # read list of datasets from first file in list (we already made sure that they are consistent between files)
            with h5py.File(os.path.join(src_dir, files[0] + ".h5"), 'r') as ref_file:
                datasets = list(ref_file[g].keys())

            for ds in datasets:
                n_events, shape, dtype, events_dim = get_dataset_properties(files, g, ds, src_dir)
                # add number of total events back into shape (along events-dimension)
                shape.insert(events_dim, sum(n_events))

                # Initialize layout for all the datasets and link it to files
                layout = h5py.VirtualLayout(shape=tuple(shape), dtype=dtype)
                for n, f in enumerate(files):
                    with h5py.File(os.path.join(src_dir, f + ".h5"), 'r') as src_file:
                        start_ind = 0 if n == 0 else sum(n_events[:n])
                        if events_dim == 0:
                            layout[start_ind:(start_ind+n_events[n]),...] = h5py.VirtualSource(path_or_dataset=src_file[g][ds])
                        else: 
                            layout[:,start_ind:(start_ind+n_events[n]),...] = h5py.VirtualSource(path_or_dataset=src_file[g][ds])

                # save virtual dataset
                current_group.create_virtual_dataset(name=ds, layout=layout)

    # Include groups from groups_include (i.e. the ones that only have to be present in one of the files and the are not
    # appended)
    with h5py.File(out_path, 'a') as out_file:
        for g in groups_include:
            current_group = out_file.require_group(g)

            # start looking for the group in all files and pick the first occurrence 
            for f in files:
                with h5py.File(os.path.join(src_dir, f + ".h5"), 'r') as in_file:
                    if g in in_file.keys():
                        # write all datasets of this group into new file
                        for ds in in_file[g]:
                            layout = h5py.VirtualLayout(shape=in_file[g][ds].shape, dtype=in_file[g][ds].dtype)
                            layout[...] = h5py.VirtualSource(path_or_dataset=in_file[g][ds])
                            current_group.create_virtual_dataset(name=ds, layout=layout)
                        
                        break # move on to next g in groups_include

    print(f"Successfully combined files {files} into '{out_path}' ({sizeof_fmt(os.path.getsize(out_path))}).")

    if extend_hours:
        print(f"Calculating extended {fmt_ds('hours')} for all groups with datasets {fmt_ds('event')}, {fmt_ds('hours')}, {fmt_ds('time_s')}, {fmt_ds('time_mus')}:")
        # We use existing functionality of the event iterator to easily create the 'hours' dataset
        dh = ai.DataHandler(nmbr_channels=1) # the number of channels is irrelevant for our purposes here
        dh.set_filepath(os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0], appendix=False)

        for k in dh.keys():
            if dh.exists(f"{k}/hours") and dh.exists(f"{k}/time_s") and dh.exists(f"{k}/time_mus"):
                sec = np.array(dh.get(k, "time_s"), dtype=np.int64)
                mus = np.array(dh.get(k, "time_mus"), dtype=np.int64)
                ts = sec*int(1e6) + mus
                old_hours = np.array(dh.get(k, "hours"), dtype=np.float32)

                earliest_ts_ind = np.argmin(ts)
                start_ts = ts[earliest_ts_ind] - int(1e6*3600*old_hours[earliest_ts_ind])

                dh.set(k, hours=(ts - start_ts)/1e6/3600, overwrite_existing=True, write_to_virtual=False)

def merge_h5(fname: str,
             files: List[str],
             src_dir: str = '',
             out_dir: str = '',
             groups_merge: List[str] = ["events", "testpulses", "noise"],
             groups_include: List[str] = [],
             extend_hours: bool = True
             ):
    """
    Merges multiple HDF5 files into a single one just like :func:`combine_h5` but it actually copies the data.

    Be aware, that the files are combined in the order as they are specified in 'files'. If you want to make sure that they are in (temporally) increasing order, you have to sort the list accordingly.

    :param fname: The name of the output file (without the .h5 extension)
    :type fname: str
    :param files: List of HDF5 files to be combined (without the .h5 extension)
    :type files: List[str]
    :param src_dir: The directory of the HDF5 files you wish to combine. Default: current directory
    :type src_dir: str
    :param out_dir: The directory where the output HDF5 file will be saved. Default: current directory
    :type out_dir: str
    :param groups_merge: Groups in the HDF5 files you wish to combine. The function will loop through all the datasets within that group and combine them along the first dimension (for 1-dimensional data), or along the second dimension (for 2- and 3-dimensional data). This is because the datasets have shape (events, data) or (channels, events, data) and we want to append them along the events-dimension.
    :type groups_merge: List[str]
    :param groups_include: Groups you just wish to copy from one representative file, i.e. the data will not be appended. This can be useful for SEVs or optimum filter, etc. 
    :type groups_include: List[str]
    :param extend_hours: If True, the ``hours`` dataset of all groups is updated in the final file such that it does not restart at 0 after every file but continuously increases. This requires the existence of datasets ``event``, ``time_s``, and ``time_mus`` in the respective groups.
    :type extend_hours: bool
    """

    # temporarily combine files (with virtual datasets)
    in_path = os.path.join(out_dir, fname + "_temp.h5")
    combine_h5(fname=fname+"_temp", files=files, src_dir=src_dir, out_dir=out_dir, groups_combine=groups_merge, groups_include=groups_include, extend_hours=extend_hours)

    # set up merged file
    out_path = os.path.join(out_dir, fname + ".h5")

    if os.path.exists(out_path):
        print(f"Overwriting existing file '{out_path}'.")
        os.remove(out_path)

    with h5py.File(in_path, 'r') as in_file:
        with h5py.File(out_path, 'a') as out_file:
            for group in in_file.keys():
                current_group = out_file.require_group(group)
                for dataset in in_file[group].keys():
                    ds2copy = in_file[group][dataset]
                    current_ds = current_group.require_dataset(name=dataset, 
                                                               shape=ds2copy.shape, 
                                                               dtype=ds2copy.dtype)
                    current_ds[...] = ds2copy

    print(f"Successfully merged files {files} into '{out_path}' ({sizeof_fmt(os.path.getsize(out_path))}).")
    # delete temporary file
    os.remove(in_path)
    print(f"Deleted '{in_path}'.")

###########################################
##### OLD MERGE FUNCTION (DEPRECATED) #####
###########################################

@deprecated(deprecated_in='1.2.3', 
            details="Use 'cait.data.combine_h5' or 'cait.data.merge_h5', which let's you combine more than two files at a time.")
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
    Merges two HDF5 files. This function is deprecated! Use :func:`combine_h5` or :func:`merge_h5` instead.

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