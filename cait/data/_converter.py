
import uproot
import h5py
import numpy as np

def convert_h5_to_root(path_h5,
                       path_root,
                       nmbr_channels):
    """
    Convert a HDF5 file to a ROOT file

    :param path_h5: string, the path to the hdf5 file that is read
    :param path_root: string, the path to the root file that is created
    :param nmbr_channels: int, the number of channels of the module
    """

    f_h5 = h5py.File(path_h5, 'r')
    f_root = uproot.recreate(path_root)

    print('HDF5 has groups: ', f_h5.keys())

    for group in f_h5.keys():
        print('ADDING TREE: ', group)
        print('HDF5 group {} has data sets: {}'.format(group, f_h5[group].keys()))
        as_dict = {}
        data_types = {}
        for dataset in f_h5[group].keys():

            # this could create a bug if the dataset randomly has the size of nmbr_channels
            if len(f_h5[group][dataset]) == nmbr_channels:
                for i in range(nmbr_channels):
                    name = dataset + '_' + str(i)
                    data_types[name] = np.float
                    # handling of the standard event - potentially this could produce bugs in the future for other single
                    # events that are stores because size is then (channels, features) instead (channels, events, features)
                    if group == 'stdevent' or group == 'stdevent_tp':
                        print('-- ADDING BRANCH: ' + name + ', LENGTH: ' + str(1))
                        as_dict[name] = np.array(f_h5[group][dataset][i]).reshape((1, -1))
                    else:
                        print('-- ADDING BRANCH: ' + name + ', LENGTH: ' + str(len(f_h5[group][dataset][i])))
                        as_dict[name] = np.array(f_h5[group][dataset][i])
            else:
                name = dataset
                print('-- ADDING BRANCH: ' + name + ', LENGTH: ' + str(len(f_h5[group][dataset])))
                data_types[name] = np.float
                as_dict[name] = np.array(f_h5[group][dataset])
        f_root[group] = uproot.newtree(data_types)
        f_root[group].extend(as_dict)

    print('Converted HDF5 to ROOT file.')