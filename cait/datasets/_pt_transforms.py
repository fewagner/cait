# imports

import torch
import numpy as np

# classes

class RemoveOffset(object):
    """
    Remove on all events the offset
    """

    def __init__(self, keys):
        """
        Remove the offset of time series data

        :param keys: the keys in the each sample-dict from that we want to remove the offset
        :type keys: list of strings
        """
        self.keys = keys

    def __call__(self, sample):
        """
        This gets applied to every sample

        :param sample: contains the features as 1D numpy arrays
        :type sample: dictionary
        :return: contains the features as 1D arrays with no offset an all keys that are in self.keys
        :rtype: dictionary
        """
        for key in self.keys:
            sample[key] = sample[key] - np.mean(sample[key][:int(len(sample[key]) / 8)])
        return sample


class Normalize(object):
    """
    Normalize Features to given mean and std
    """

    def __init__(self, norm_vals):
        """
        Hand the values for the normalizations

        :param norm_vals: each key corresponds to a key in the sample and is a list of length two: [mean, std]
        :type norm_vals: dictionary
        """
        self.norm_vals = norm_vals

    def __call__(self, sample):
        """
        This gets applied to every sample

        :param sample: contains the features as 1D numpy arrays
        :type sample: dictionary
        :return: contains the scaled features as 1D arrays
        :rtype: dictionary
        """

        for k in self.norm_vals.keys():
            mean, std = self.norm_vals[k]
            sample[k] = (sample[k] - mean) / std

        return sample


class DownSample(object):
    """
    Sample all the time series down
    """

    def __init__(self, keys, down):
        """
        Tell initial info to the instance

        :param keys: the keys in each sample-dist we want to downsample
        :type keys: list of strings
        """
        self.keys = keys
        self.down = down

    def __call__(self, sample):
        """
        This gets applied to every sample

        :param sample: contains the features as 1D numpy arrays
        :type sample: dictionary
        :return: contains the downsampled features as 1D arrays
        :rtype: dictionary
        """
        for key in self.keys:
            sample[key] = np.mean(sample[key].
                                  reshape(int(len(sample[key]) / self.down), self.down),
                                  axis=1)
        return sample


class ToTensor(object):
    """
    Convert numpy arrays in sample to Tensors
    """

    def __call__(self, sample):
        """
        This gets applied to every sample

        :param sample: contains the features as 1D numpy arrays
        :type sample: dictionary
        :return: contains the features as 1D torch tensors
        :rtype: dictionary
        """
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])
        return sample
