from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ._pt_dataset import H5CryoData
from ._pt_sampler import get_random_samplers
import torch.nn.functional as F


class CryoDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for processing of HDF5 dataset
    """

    def __init__(self, hdf5_path, type, keys, channel_indices, transform=None, nmbr_events=None):
        super().__init__()
        """
        Give instructions how to extract the data from the h5 set

        :param hdf5_path: string, full path to the hdf5 data set
        :param type: string, either events or testpulses or noise - the group index of the hd5 data set
        :param keys: list of strings, the keys that are accessed in the hdf5 group
        :param channel_indices: list of lists or Nones, must have same length than the keys list, the channel indices
            of the data sets in the group, if None then no index is set (i.e. if the h5 data set does not belong to
            a specific channel)
        :param transform: pytorch transforms class, get applied to every sample when getitem is called
        :param nmbr_events: int or None, if set this is the number of events in the data set, if not it is extracted
            from the hdf5 file with len(f['events/event'][0])
        :param nmbr_channels: int or None, if set this is the number of channels we want to include in the pytorch
            data set, if not it is extracted from the h5 file with with len(f['events/event'])
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.type = type
        self.keys = keys
        self.channel_indices = channel_indices
        self.transform = transform
        self.nmbr_events = nmbr_events

    def prepare_data(self, val_size, test_size, batch_size, dataset_size=None, only_idx=None, shuffle_dataset=True, random_seed=None,
                     feature_keys=[], label_keys=[], one_hot=False):
        # called only on 1 GPU
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.only_idx = only_idx
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        self.feature_keys = feature_keys
        self.label_keys = label_keys
        self.one_hot = one_hot

    def setup(self):
        # called on every GPU
        self.dataset_full = H5CryoData(hdf5_path=self.hdf5_path,
                                  type=self.type,
                                  keys=self.keys,
                                  channel_indices=self.channel_indices,
                                  transform=self.transform,
                                  nmbr_events=self.nmbr_events)

        if self.dataset_size is None:
            self.dataset_size = len(self.dataset_full)

        self.train_sampler, self.val_sampler, self.test_sampler = get_random_samplers(test_size=self.test_size,
                                                                           val_size=self.val_size,
                                                                           dataset_size=self.dataset_size,
                                                                           only_idx=self.only_idx,
                                                                           shuffle_dataset=self.shuffle_dataset,
                                                                           random_seed=self.random_seed)

        # now get the number of samples and the number of features per sample
        # this is consistent with downsampled time series :)
        if self.only_idx is None:
            nmbr_samples = self.dataset_size
        else:
            nmbr_samples = len(self.only_idx)

        sample_keys = self.dataset_full[0].keys()

        nmbr_features = 0
        for k in sample_keys:
            if k in self.feature_keys:
                nmbr_features += len(self.dataset_full[0][k])

        nmbr_labels = 0
        for k in sample_keys:
            if k in self.label_keys:
                if self.one_hot:
                    nmbr_labels += len(F.one_hot(self.dataset_full[0][k]))
                else:
                    nmbr_labels += len(self.dataset_full[0][k])

        self.dims = (nmbr_samples, nmbr_features)  # returns full length of data set and nmbr of features
        self.label_dims = (nmbr_samples, nmbr_labels)

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_full, batch_size=batch_size, sampler=self.train_sampler)

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_full, batch_size=batch_size, sampler=self.val_sampler)

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_full, batch_size=batch_size, sampler=self.test_sampler)
