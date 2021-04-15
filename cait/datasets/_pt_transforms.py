# imports

import torch
import numpy as np
import torch.nn.functional as F

# classes

class RemoveOffset(object):
    """
    Remove on all events the offset

    :param keys: the keys in the each sample-dict from that we want to remove the offset
    :type keys: list of strings
    """

    def __init__(self, keys):
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

    :param norm_vals: each key corresponds to a key in the sample and is a list of length two: [mean, std],
            or if type = 'minmax' then [min, max]
    :type norm_vals: dictionary
    :param type: 'z' for calculating Z-scores or 'minmax' of scaling from 0 to 1
    :type type: string
    """

    def __init__(self, norm_vals, type='z'):
        self.norm_vals = norm_vals
        self.type = type

    def __call__(self, sample):
        """
        This gets applied to every sample

        :param sample: contains the features as 1D numpy arrays
        :type sample: dictionary
        :return: contains the scaled features as 1D arrays
        :rtype: dictionary
        """

        if self.type == 'z':
            for k in self.norm_vals.keys():
                mean, std = self.norm_vals[k]
                sample[k] = (sample[k] - mean) / std
        elif self.type == 'minmax':
            for k in self.norm_vals.keys():
                min, max = self.norm_vals[k]
                sample[k] = (sample[k] - min) / (max - min)

        return sample


class DownSample(object):
    """
    Sample all the time series down

    :param keys: the keys in each sample-dist we want to downsample
    :type keys: list of strings
    """

    def __init__(self, keys, down):
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
            sample[key] = torch.from_numpy(sample[key]).float()
        return sample


class OneHotEncode(object):
    """
    One Hot Encode the label incides

    :param keys: the keys in each sample-dist we want to one hot encode
    :type keys: list of strings
    :param nmbr_classes: the number of classes to that we one hot encode
    :type nmbr_classes: int
    """

    def __init__(self, keys, nmbr_classes):
        self.keys = keys
        self.nmbr_classes = nmbr_classes

    def __call__(self, sample):
        """
        This gets applied to every sample

        :param sample: contains the features as 1D numpy arrays
        :type sample: dictionary
        :return: contains the features as 1D arrays with the keys in self.keys one hot encoded
        :rtype: dictionary
        """
        for key in self.keys:
            sample[key] = F.one_hot(sample[key].long(), num_classes=self.nmbr_classes)
        return sample