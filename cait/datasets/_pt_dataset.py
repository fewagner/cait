
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn.functional as F

class H5CryoData(Dataset):
    """
    Pytorch Dataset for the processing of raw data from hdf5 files
    """

    def __init__(self, hdf5_path, type, keys, channel_indices,
                 keys_one_hot=[], transform=None, nmbr_events=None):
        """
        Give instructions how to extract the data from the h5 set

        :param hdf5_path: string, full path to the hdf5 data set
        :param type: string, either events or testpulses or noise - the group index of the hd5 data set
        :param keys: list of strings, the keys that are accessed in the hdf5 group
        :param channel_indices: list of lists or Nones, must have same length than the keys list, the channel indices
            of the data sets in the group, if None then no index is set (i.e. if the h5 data set does not belong to
            a specific channel)
        :param keys_one_hot: list of strings, the keys that get one hot encoded - important for correct size
        :param transform: pytorch transforms class, get applied to every sample when getitem is called
        :param nmbr_events: int or None, if set this is the number of events in the data set, if not it is extracted
            from the hdf5 file with len(self.f['events/event'][0])
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.f = h5py.File(hdf5_path, 'r')
        self.type = type
        self.keys = keys
        self.channel_indices = channel_indices
        self.keys_one_hot = keys_one_hot
        if nmbr_events == None:
            self.nmbr_events = len(self.f['events/event'][0])
        else:
            self.nmbr_events = nmbr_events
        for chs in channel_indices:
            if chs is not None:
                self.nmbr_channels = len(chs)
                break

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

        for i, key in enumerate(self.keys): # all the elements of the dict have size (nmbr_features)
            if self.channel_indices[i] is not None:
                for c in self.channel_indices[i]:
                    new_key = key + '_ch' + str(c)
                    sample[new_key] = np.array(self.f[self.type][key][c, idx])
                    if new_key in self.keys_one_hot:
                        pass # one hot encoding is now done in a transform
                    elif len(sample[new_key].shape) == 0:  # must have len dim = 1
                        sample[new_key] = np.array(self.f[self.type][key][c, idx]).reshape(1)
            else:
                sample[key] = np.array(self.f[self.type][key][idx])
                if key in self.keys_one_hot:
                    pass  # one hot encoding is now done in a transform
                elif len(sample[key].shape) == 0:
                    sample[key] = sample[key].reshape(1)

        if self.transform:
            sample = self.transform(sample)

        # final check if dimensions are alright
        for k in sample.keys():
            if len(sample[k].shape) != 1 and k not in self.keys_one_hot:
                raise KeyError('The arrays in the dict-samples in H5CryoData must all have dim=1.')

        return sample