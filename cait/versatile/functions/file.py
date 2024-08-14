import os
import h5py
from typing import List

import cait as ai
from ...styles._print_styles import sizeof_fmt, fmt_ds

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
        
def combine(fname: str, 
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
            if dh.exists(f"{k}/hours") and dh.exists(f"{k}/event") and dh.exists(f"{k}/time_s") and dh.exists(f"{k}/time_mus"):
                dh.set(k, hours=dh.get_event_iterator(k).hours, overwrite_existing=True, write_to_virtual=False)

def merge(fname: str, 
          files: List[str], 
          src_dir: str = '', 
          out_dir: str = '', 
          groups_merge: List[str] = ["events", "testpulses", "noise"], 
          groups_include: List[str] = [],
          extend_hours: bool = True):
    """
    Merges multiple HDF5 files into a single one just like `func:combine` but it actually copies the data.

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
    combine(fname=fname+"_temp", files=files, src_dir=src_dir, out_dir=out_dir, groups_combine=groups_merge, groups_include=groups_include, extend_hours=extend_hours)

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