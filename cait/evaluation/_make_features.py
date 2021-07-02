import numpy as np


def make_features(X):
    """
    Takes a list of chosen features and attaches them to one another.

    :param X: The features we want to attach to each other.
    :type X: list of arrays of same length
    """
    for i, x in enumerate(X):
        # want to convert 1D arrays to shape (n,1)
        if x.ndim == 1:
            X[i] = x[:, None]

    # glue together ist of features and return them
    return np.hstack([x for x in X])