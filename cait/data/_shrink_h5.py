
import h5py
import numpy as np

def shrink_h5(path_original, path_new,
              cutflag, type='events',):
    """
    Create a new dataset that holds only events that are not cut from the original dataset.

    :param path_original: The full path to the original HDF5 set.
    :type path_original: str
    :param path_new: The full path where the new HDF5 set is created.
    :type path_new: str
    :param cutflag: The cutflag corresponding to the events in the original data set.
    :type cutflag: 1D bool array
    :param type: The group name in the original HDF5 data set. Typically "events", "testpulses" or "noise".
    :type type: str
    """

    len_cutflag = cutflag.shape[0]
    goods_cutflag = np.sum(cutflag)
    idx_goods = cutflag.nonzero()[0]

    with h5py.File(path_original, 'r') as f_original, h5py.File(path_new, 'w') as f_new:
        for group in list(f_original.keys()):
            if group == type:
                f_new.create_group(group)
                print('shrink group ', group)
                for set in f_original[type].keys():
                    shape = list(f_original[type][set].shape)
                    shrink_dim = None
                    for i, d in enumerate(shape):
                        if d == len_cutflag:
                            shrink_dim = i
                            shape[i] = goods_cutflag
                            print(f'shrink set {set} dim {i} from {len_cutflag} to {goods_cutflag}')
                            break
                    f_new[group].create_dataset(set,
                                         shape=shape,
                                         dtype=f_original[group][set].dtype)
                    if shrink_dim is None:
                        f_new[group][set][...] = f_original[group][set][...]
                    elif shrink_dim == 0:
                        f_new[group][set][:, ...] = f_original[group][set][idx_goods, ...]
                    elif shrink_dim == 1:
                        f_new[group][set][:, :, ...] = f_original[group][set][:, idx_goods, ...]
                    elif shrink_dim == 2:
                        f_new[group][set][:, :, :, ...] = f_original[group][set][:, :, idx_goods, ...]
                    elif shrink_dim == 3:
                        f_new[group][set][:, :, :, :, ...] = f_original[group][set][:, :, :, idx_goods, ...]
                    else:
                        raise NotImplementedError('data sets with more than 4 dimensions are currently not supported!')

            else:
                print('copy group ', group)
                f_original.copy(group, f_new)

