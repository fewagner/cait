from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ._pt_dataset import H5CryoData
from ._pt_sampler import get_random_samplers
import torch.nn.functional as F


class CryoDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for processing of HDF5 dataset
    """

    def __init__(self, hdf5_path, type, keys, channel_indices, feature_indices,
                 transform=None, nmbr_events=None):
        super().__init__()
        """
        Give instructions how to extract the data from the h5 set

        :param hdf5_path: string, full path to the hdf5 data set
        :param type: string, either events or testpulses or noise - the group index of the hd5 data set
        :param keys: list of strings, the keys that are accessed in the hdf5 group
        :param channel_indices: list of lists or Nones, must have same length than the keys list, the channel indices
            of the data sets in the group, if None then no index is set (i.e. if the h5 data set does not belong to
            a specific channel)
        :param feature_indices: list of lists or Nones, must have same length than the keys list, the feature indices
            of the data sets in the group (third idx),
            if None then no index is set (i.e. there is no third index in the set or all features are chosen)
        :param transform: pytorch transforms class, get applied to every sample when getitem is called
        :param nmbr_events: int or None, if set this is the number of events in the data set, if not it is extracted
            from the hdf5 file with len(f['events/event'][0])
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.type = type
        self.keys = keys
        self.channel_indices = channel_indices
        self.feature_indices = feature_indices
        self.transform = transform
        self.nmbr_events = nmbr_events

    def prepare_data(self, val_size, test_size, batch_size, nmbr_workers, load_to_memory=False,
                     dataset_size=None, only_idx=None,
                     shuffle_dataset=True, random_seed=None,
                     feature_keys=[], label_keys=[], keys_one_hot=[]):
        """
        Called once to hand additional info about the data setup, info for training

        :param val_size: the size of the validation set
        :type val_size: float between 0 and 1
        :param test_size: the size of the test set
        :type test_size:  float between 0 and 1
        :param batch_size: the batch size in the training process
        :type batch_size: int
        :param nmbr_workers: the number of processes to run, best choose the number of CPUs on the machine - this might
            cause issues if load_to_memory is not activated
        :type nmbr_workers: int
        :param load_to_memory: if set, the whole data gets loaded into memory, if nmbr_workers > 0 this is recommended
        :type load_to_memory: bool
        :param dataset_size: the size of the whole dataset, gets overwritten if only_idx is set
        :type dataset_size: int or None
        :param only_idx: only these indices are then used from the initial dataset/h5 file
        :type only_idx: list of ints or None
        :param shuffle_dataset: the train set gets shuffled after every epoch
        :type shuffle_dataset: bool
        :param random_seed: if we want to use a random seed to reproduce the results
        :type random_seed: int or None
        :param feature_keys: data from these keys is supposed to be input to the NN
        :type feature_keys: list of strings
        :param label_keys: data from these keys is supposed to be labels for the NN training
        :type label_keys: list of strings
        :param keys_one_hot: this data gets one-hot encoded
        :type channels_one_hot: list of strings
        :return: -
        :rtype: -
        """
        # called only on 1 GPU
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.nmbr_workers = nmbr_workers
        self.only_idx = only_idx
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        self.feature_keys = feature_keys
        self.label_keys = label_keys
        self.keys_one_hot = keys_one_hot
        self.load_to_memory = load_to_memory

        if not load_to_memory and nmbr_workers > 0:
            print('Attention: nmbr_workers > 0 and not load to memory might cause issues with the h5 file read!')

        self.dataset_full = H5CryoData(hdf5_path=self.hdf5_path,
                                       type=self.type,
                                       keys=self.keys,
                                       channel_indices=self.channel_indices,
                                       feature_indices=self.feature_indices,
                                       keys_one_hot=self.keys_one_hot,
                                       transform=self.transform,
                                       nmbr_events=self.nmbr_events,
                                       load_to_memory=self.load_to_memory)

    def setup(self):
        """
        Called on every GPU before start of training, here creation of dataset and splits in samplers are done

        :return: -
        :rtype: -
        """

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

        nmbr_features = 0
        for k in self.feature_keys:
            nmbr_features += len(self.dataset_full[0][k])

        self.dims = (nmbr_samples, nmbr_features)  # returns full length of data set and nmbr of features

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_full, batch_size=batch_size, sampler=self.train_sampler,
                          num_workers=self.nmbr_workers)

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_full, batch_size=batch_size, sampler=self.val_sampler,
                          num_workers=self.nmbr_workers)

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_full, batch_size=batch_size, sampler=self.test_sampler,
                          num_workers=self.nmbr_workers)
