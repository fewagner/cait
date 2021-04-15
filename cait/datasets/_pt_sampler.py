import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def get_random_samplers(test_size, val_size, dataset_size=None, only_idx=None, shuffle_dataset=True, random_seed=None):
    """
    Chooses the indices for the Split datasets.

    :param test_size: float between 0 and 1, the size of the testset
    :param val_size: float between 0 and 1, the size of the validation set
    :param dataset_size: Size of the whole dataset, is a number
    :param only_idx: list of ints or None, if set only these indices from the dataset are included
    :param shuffle_dataset: When true, the indices are dataset is shuffled befor the indices are assigned
    :param random_seed: set of some value to get the same datasets always for comparability
    :return: indices for training, validation and test set
    """

    if dataset_size is None and only_idx is None:
        raise KeyError('At least one of dataset_size and only_idx must be set!')

    if only_idx is None:
        only_idx = list(range(dataset_size))
    val_split = int((1 - (val_size + test_size)) * len(only_idx))  # floor rounds down
    test_split = int((1 - (test_size)) * len(only_idx))
    if shuffle_dataset:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(only_idx)
    train_indices, val_indices, test_indices = only_idx[:val_split], only_idx[val_split:test_split], only_idx[test_split:]

    return SubsetRandomSampler(train_indices), \
           SubsetRandomSampler(val_indices), \
           SubsetRandomSampler(test_indices)