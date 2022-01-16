from torch.utils.data import DataLoader, BatchSampler
import pytorch_lightning as pl
from ._pt_dataset import H5CryoData
from ._pt_sampler import get_random_samplers
import torch.nn.functional as F
from ._pt_dataloader import FastDataLoader
import warnings


class CryoDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for processing of HDF5 dataset.

    :param hdf5_path: Full path to the hdf5 data set.
    :type hdf5_path: string
    :param type: Either events or testpulses or noise - the group index of the hd5 data set.
    :type type: string
    :param keys: The keys that are accessed in the hdf5 group.
    :type keys: list of strings
    :param channel_indices: Must have same length than the keys list, the channel indices
        of the data sets in the group. If None then no index is set (i.e. if the h5 data set does not belong to
        a specific channel).
    :type channel_indices: list of lists or Nones
    :param feature_indices: Must have same length than the keys list, the feature indices
        of the data sets in the group (third idx).
        If None then no index is set (i.e. there is no third index in the set or all features are chosen)
    :type feature_indices: list of lists or Nones
    :param transform: Get applied to every sample when getitem is called.
    :type transform: pytorch transforms class
    :param nmbr_events: If set this is the number of events in the data set, if not it is extracted
        from the hdf5 file with len(f['events/event'][0]).
    :type nmbr_events: int or None
    :param double: If true all events are cast to double before calculations.
    :type double: bool
    """

    def __init__(self, hdf5_path, type, keys, channel_indices, feature_indices=None,
                 transform=None, nmbr_events=None, double=False):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.type = type
        self.keys = keys
        self.channel_indices = channel_indices
        self.feature_indices = feature_indices
        self.transform = transform
        self.nmbr_events = nmbr_events
        self.double = double

    def prepare_data(self, val_size, test_size, batch_size, nmbr_workers, load_to_memory=False,
                     dataset_size=None, only_idx=None,
                     shuffle_dataset=True, random_seed=None,
                     feature_keys=[], label_keys=[], keys_one_hot=[],
                     ):
        """
        Called once to hand additional info about the data setup.

        :param val_size: The size of the validation set.
        :type val_size: float between 0 and 1
        :param test_size: The size of the test set.
        :type test_size:  float between 0 and 1
        :param batch_size: The batch size in the training process.
        :type batch_size: int
        :param nmbr_workers: The number of processes to run, best choose the number of CPUs on the machine - this might
            cause issues if load_to_memory is not activated.
        :type nmbr_workers: int
        :param load_to_memory: Depricated! Not recommended! If set, the whole data gets loaded into memory.
        :type load_to_memory: bool
        :param dataset_size: The size of the whole dataset, gets overwritten if only_idx is set.
        :type dataset_size: int or None
        :param only_idx: Only these indices are then used from the initial dataset/h5 file.
        :type only_idx: list of ints or None
        :param shuffle_dataset: The train set gets shuffled after every epoch.
        :type shuffle_dataset: bool
        :param random_seed: If we want to use a random seed to reproduce the results.
        :type random_seed: int or None
        :param feature_keys: Data from these keys is supposed to be input to the NN.
        :type feature_keys: list of strings
        :param label_keys: Data from these keys is supposed to be labels for the NN training.
        :type label_keys: list of strings
        :param keys_one_hot: This data gets one-hot encoded.
        :type keys_one_hot: list of strings
        """
        # called only on 1 worker
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
        # self.feature_dims = feature_dims

        if load_to_memory:
            warnings.warn('Attention: The feature load_to_memory is depricated and not recommended!')

        # if not load_to_memory and nmbr_workers > 0:
        #     warnings.warn('Attention: nmbr_workers > 0 and not load to memory might cause issues with the h5 file read!')

        self.dataset_full = H5CryoData(hdf5_path=self.hdf5_path,
                                       type=self.type,
                                       keys=self.keys,
                                       channel_indices=self.channel_indices,
                                       feature_indices=self.feature_indices,
                                       keys_one_hot=self.keys_one_hot,
                                       transform=self.transform,
                                       nmbr_events=self.nmbr_events,
                                       double=self.double,
                                       # feature_dims=self.feature_dims,
                                       )

    def setup(self):
        """
        Called on every worker before start of training, here creation of dataset and splits in samplers are done.
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
            nmbr_features += len(self.dataset_full.get_item_no_cache(0)[k])

        self.dims = (nmbr_samples, nmbr_features)  # returns full length of data set and nmbr of features

    def train_dataloader(self, batch_size=None):
        """
        Return the training data loader.

        :param batch_size: The batchsize.
        :type batch_size: int
        :return: Instance of FastDataLoader, a child of the PyTorch DataLoader, developed from within the
            PyTorch community.
        :rtype: object
        """
        if batch_size is None:
            batch_size = self.batch_size
        return FastDataLoader(self.dataset_full,
                              batch_sampler=BatchSampler(self.train_sampler, batch_size, drop_last=False),
                              num_workers=self.nmbr_workers)

    def val_dataloader(self, batch_size=None):
        """
        Return the validation data loader.

        :param batch_size: The batchsize.
        :type batch_size: int
        :return: Instance of FastDataLoader, a child of the PyTorch DataLoader, developed from within the
            PyTorch community.
        :rtype: object
        """
        if batch_size is None:
            batch_size = self.batch_size
        return FastDataLoader(self.dataset_full,
                              batch_sampler=BatchSampler(self.val_sampler, batch_size, drop_last=False),
                              num_workers=self.nmbr_workers)

    def test_dataloader(self, batch_size=None):
        """
        Return the test data loader.

        :param batch_size: The batchsize.
        :type batch_size: int
        :return: Instance of FastDataLoader, a child of the PyTorch DataLoader, developed from within the
            PyTorch community.
        :rtype: object
        """
        if batch_size is None:
            batch_size = self.batch_size
        return FastDataLoader(self.dataset_full,
                              batch_sampler=BatchSampler(self.test_sampler, batch_size, drop_last=False),
                              num_workers=self.nmbr_workers)
