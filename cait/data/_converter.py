
import uproot
import h5py
import numpy as np

def convert_h5_to_root(path_h5,
                       path_root):
    """
    Convert a HDF5 file to a ROOT file

    :param path_h5: string, the path to the hdf5 file that is read
    :param path_root: string, the path to the root file that is created
    """

    f_h5 = h5py.File(path_h5, 'r')
    f_root = uproot.recreate(path_root)

    for group in f_h5.keys():
        as_dict = {}
        data_types = {}
        for dataset in f_h5[group].keys():
            data_types[dataset] = np.float
            as_dict[dataset] = np.array(f_h5[group][dataset])
        f_root[group] = uproot.newtree(data_types)
        f_root[group].extend(as_dict)

    print('Converted HDF5 to ROOT file.')