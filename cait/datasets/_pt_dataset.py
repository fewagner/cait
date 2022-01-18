from torch.utils.data import Dataset
import h5py
import numpy as np
import torch


class H5CryoData(Dataset):
    """
    Pytorch Dataset for the processing of raw data from hdf5 files with a Cait-like file structure.

    :param type: Either events or testpulses or noise - the group index of the hd5 data set.
    :type type: string,
    :param keys: The keys that are accessed in the hdf5 group.
    :type keys: list of strings
    :param channel_indices: Must have same length than the keys list, the channel indices
        of the data sets in the group. If None then no index is set (i.e. if the h5 data set does not belong to
        a specific channel).
    :type channel_indices: list of lists or Nones
    :param feature_indices: Must have same length than the keys list, the feature indices
        of the data sets in the group (third idx).
        If None then no index is set (i.e. there is no third index in the set or all features are chosen).
    :type feature_indices: list of lists or Nones
    :param keys_one_hot: The keys that get one hot encoded - important for correct size.
    :type keys_one_hot: list of strings
    :param hdf5_path: Full path to the hdf5 data set, need be provided if no file handle is set.
    :type hdf5_path: string or None
    :param transform: Get applied to every sample when getitem is called.
    :type transform: pytorch transforms class
    :param nmbr_events: If set this is the number of events in the data set, if not it is extracted
        from the hdf5 file with len(self.f['events/event'][0]).
    :type nmbr_events: int or None
    :param double: If true all events are cast to double before calculations.
    :type double: bool
    """

    def __init__(self, type, keys, channel_indices, feature_indices=None,
                 keys_one_hot=[], hdf5_path=None,
                 transform=None, nmbr_events=None, double=False,
                 ):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.type = type
        self.keys = keys
        self.channel_indices = channel_indices
        if feature_indices is None:
            self.feature_indices = [None for i in range(len(keys))]
        else:
            self.feature_indices = feature_indices
        self.keys_one_hot = keys_one_hot
        for chs in channel_indices:
            if chs is not None:
                self.nmbr_channels = len(chs)
                break
        if nmbr_events == None:
            with h5py.File(hdf5_path, 'r') as f:
                self.nmbr_events = len(f[type + '/event'][0])
        else:
            self.nmbr_events = nmbr_events
        self.double = double
        self.h5_set = None

    def __len__(self):
        """
        Returns the number of events in the data set.
        """
        return self.nmbr_events

    def get_item_no_cache(self, idx):
        """
        Returns the sample of the dataset at idx, without caching the file stream.

        :param idx: The index at which we want to get the event.
        :type idx: int
        :return: Each element is a numpy array.
        :rtype: dict
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            sample = self.build_sample(f, idx)

        if self.transform:
            sample = self.transform(sample)

        for k in sample.keys():
            if self.double:
                sample[k] = sample[k].double()

        return sample

    def build_sample(self, f, idx):
        """
        Build a sample that is handed to the pytorch model.

        :param f: The file handle to the HDF5 data set.
        :type f: hdf5 file stream
        :param idx: The index of the event to process from the HDF5 data set.
        :type idx: int
        :return: A dictionary with the data from the HDF5 data set.
        :rtype: dict
        """
        sample = {}
        for i, key in enumerate(self.keys):  # all the elements of the dict have size (nmbr_features)
            ls = len(f[self.type + '/' + key].shape)
            if ls == 1 and self.channel_indices[i] is None and self.feature_indices[i] is None:
                if key not in self.keys_one_hot:
                    sample[key] = np.array(f[self.type + '/' + key][idx]).reshape(1)  # e.g. true onset, ...
                else:
                    sample[key] = np.array(f[self.type + '/' + key][idx])
            elif ls == 2 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                for c in self.channel_indices[i]:
                    new_key = key + '_ch' + str(c)
                    if new_key not in self.keys_one_hot:  # e.g. labels
                        sample[new_key] = np.array(f[self.type + '/' + key][c, idx]).reshape(1)  # e.g. true ph, ...
                    else:
                        sample[new_key] = np.array(f[self.type + '/' + key][c, idx])
            elif ls == 3 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                for c in self.channel_indices[i]:
                    new_key = key + '_ch' + str(c)
                    sample[new_key] = np.array(f[self.type + '/' + key][c, idx])  # e.g. event, ...
            elif ls == 3 and self.channel_indices[i] is not None and self.feature_indices[i] is not None:
                for c in self.channel_indices[i]:
                    for fe in self.feature_indices[i]:
                        new_key = key + '_ch' + str(c) + '_fe' + str(fe)
                        if new_key not in self.keys_one_hot:
                            sample[new_key] = np.array(f[self.type + '/' + key][c, idx, fe]).reshape(
                                1)  # e.g. single mp, ...
                        else:
                            sample[new_key] = np.array(f[self.type + '/' + key][c, idx, fe])
            elif ls == 4 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                for c in self.channel_indices[i]:
                    new_key = key + '_ch' + str(c)
                    sample[new_key] = np.array(f[self.type + '/' + key][c, idx])  # e.g. pulse_traces, ...
            else:
                raise KeyError('For {} the combination of channel_indices and feature_indices is invalid.'.format(key))
        return sample

    def __getitem__(self, idx):
        """
        Returns the sample of the dataset at idx.

        :param idx: The index at which we want to get the event.
        :type idx: int
        :return: Each element is a numpy array.
        :rtype: dict
        """

        if self.h5_set is None:
            self.h5_set = {}
            for k in self.keys:
                self.h5_set[self.type + '/' + k] = h5py.File(self.hdf5_path, 'r')[self.type + '/' + k]
        sample = self.build_sample(self.h5_set, idx)

        if self.transform:
            sample = self.transform(sample)

        for k in sample.keys():
            if self.double:
                sample[k] = sample[k].double()

        return sample
