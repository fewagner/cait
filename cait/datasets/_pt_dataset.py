
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch

class H5CryoData(Dataset):
    """
    Pytorch Dataset for the processing of raw data from hdf5 files
    """

    def __init__(self, type, keys, channel_indices, feature_indices,
                 keys_one_hot=[], hdf5_path = None, file_handle=None,
                 transform=None, nmbr_events=None, load_to_memory=False, double=False):
        """
        Give instructions how to extract the data from the h5 set

        :param hdf5_path: string or None, full path to the hdf5 data set, need be provided if no file handle is set
        :param file_handle: hdf5 file stream or None, the stream of the h5 file that holds the data
        :param type: string, either events or testpulses or noise - the group index of the hd5 data set
        :param keys: list of strings, the keys that are accessed in the hdf5 group
        :param channel_indices: list of lists or Nones, must have same length than the keys list, the channel indices
            of the data sets in the group, if None then no index is set (i.e. if the h5 data set does not belong to
            a specific channel)
        :param feature_indices: list of lists or Nones, must have same length than the keys list, the feature indices
            of the data sets in the group (third idx),
            if None then no index is set (i.e. there is no third index in the set or all features are chosen)
        :param keys_one_hot: list of strings, the keys that get one hot encoded - important for correct size
        :param transform: pytorch transforms class, get applied to every sample when getitem is called
        :param nmbr_events: int or None, if set this is the number of events in the data set, if not it is extracted
            from the hdf5 file with len(self.f['events/event'][0])
        :param load_to_memory: bool, if set the whole data gets loaded into memory when the dataset is created - causes
            less problems with multiprocessing but might cause memory issues
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.type = type
        self.keys = keys
        self.channel_indices = channel_indices
        self.feature_indices = feature_indices
        self.keys_one_hot = keys_one_hot
        for chs in channel_indices:
            if chs is not None:
                self.nmbr_channels = len(chs)
                break
        self.load_to_memory = load_to_memory
        if load_to_memory:
            if file_handle is None:
                with h5py.File(hdf5_path, 'r') as file_handle:
                    self.data = self.load_data(file_handle)
            else:
                self.data = self.load_data(file_handle)
        else:
            pass  # h5 file is only opened in the __getitem__ method!
        if nmbr_events == None:
            if load_to_memory:
                self.nmbr_events = len(self.data[list(self.data.keys())[0]])
            else:
                with h5py.File(hdf5_path, 'r') as f:
                    self.nmbr_events = len(f[type + '/event'][0])
        else:
            self.nmbr_events = nmbr_events
        self.double = double
        self.file_handle = file_handle

    def load_data(self, f):

        data = {}

        for i, key in enumerate(self.keys):  # all the elements of the dict have size (nmbr_features)
            ls = len(f[self.type][key].shape)
            if ls == 1 and self.channel_indices[i] is None and self.feature_indices[i] is None:
                if key not in self.keys_one_hot:
                    data[key] = np.array(f[self.type][key]).reshape(-1, 1)  # e.g. true onset, ...
                else:
                    data[key] = np.array(f[self.type][key])
            elif ls == 2 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                for c in self.channel_indices[i]:
                    new_key = key + '_ch' + str(c)
                    if new_key not in self.keys_one_hot:  # e.g. labels
                        data[new_key] = np.array(f[self.type][key][c]).reshape(-1, 1)  # e.g. true ph, ...
                    else:
                        data[new_key] = np.array(f[self.type][key][c])
            elif ls == 3 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                for c in self.channel_indices[i]:
                    new_key = key + '_ch' + str(c)
                    data[new_key] = np.array(f[self.type][key][c])  # e.g. event, ...
            elif ls == 3 and self.channel_indices[i] is not None and self.feature_indices[i] is not None:
                for c in self.channel_indices[i]:
                    for fe in self.feature_indices[i]:
                        new_key = key + '_ch' + str(c) + '_fe' + str(fe)
                        if new_key not in self.keys_one_hot:
                            data[new_key] = np.array(f[self.type][key][c, :, fe]).reshape(-1, 1)  # e.g. single mp, ...
                        else:
                            data[new_key] = np.array(f[self.type][key][c, :, fe])
            else:
                raise KeyError('For {} the combination of channel_indices and feature_indices is invalid.'.format(key))

        return data

    def __len__(self):
        """
        Returns the number of events in the data set
        """
        return self.nmbr_events

    def __getitem__(self, idx):
        """
        Returns the sample of the dataset at idx

        :param idx: int, the index at which we want to get the event
        :return: dict, each element is a numpy array
        """

        sample = {}

        if self.load_to_memory:
            for key in self.data.keys():
                sample[key] = self.data[key][idx]
        else:
            if self.file_handle is None:
                if self.hdf5_path is not None:
                    self.f = h5py.File(self.hdf5_path, 'r')  # this causes trouble with multiple workers
                else:
                    raise KeyError('If you provide no file handle, you must provide a hdf5 path!')
            for i, key in enumerate(self.keys):  # all the elements of the dict have size (nmbr_features)
                ls = len(self.f[self.type][key].shape)
                if ls == 1 and self.channel_indices[i] is None and self.feature_indices[i] is None:
                    if key not in self.keys_one_hot:
                        sample[key] = np.array(self.f[self.type][key][idx]).reshape(1)  # e.g. true onset, ...
                    else:
                        sample[key] = np.array(self.f[self.type][key][idx])
                elif ls == 2 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                    for c in self.channel_indices[i]:
                        new_key = key + '_ch' + str(c)
                        if new_key not in self.keys_one_hot:  # e.g. labels
                            sample[new_key] = np.array(self.f[self.type][key][c, idx]).reshape(1)  # e.g. true ph, ...
                        else:
                            sample[new_key] = np.array(self.f[self.type][key][c, idx])
                elif ls == 3 and self.channel_indices[i] is not None and self.feature_indices[i] is None:
                    for c in self.channel_indices[i]:
                        new_key = key + '_ch' + str(c)
                        sample[new_key] = np.array(self.f[self.type][key][c, idx])  # e.g. event, ...
                elif ls == 3 and self.channel_indices[i] is not None and self.feature_indices[i] is not None:
                    for c in self.channel_indices[i]:
                        for fe in self.feature_indices[i]:
                            new_key = key + '_ch' + str(c) + '_fe' + str(fe)
                            if new_key not in self.keys_one_hot:
                                sample[new_key] = np.array(self.f[self.type][key][c, idx, fe]).reshape(1)  # e.g. single mp, ...
                            else:
                                sample[new_key] = np.array(self.f[self.type][key][c, idx, fe])
                else:
                    raise KeyError('For {} the combination of channel_indices and feature_indices is invalid.'.format(key))

        if self.transform:
            sample = self.transform(sample)

        # final check if dimensions are alright
        for k in sample.keys():
            if len(sample[k].shape) != 1 and k not in self.keys_one_hot:
                raise KeyError('The {} must have dim=1 but has dim={}. If it is a label, put in keys_one_hot.'.format(k, len(sample[k].shape)))
            if self.double:
                sample[k] = sample[k].double()

        return sample