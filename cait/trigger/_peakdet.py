import numpy as np
import numba as nb

@nb.njit
def add_to_moments(x_new, mean, var, n):
    """
    Update existing mean and variance values by adding a new data point.

    :param x_new:
    :type x_new:
    :param mean:
    :type mean:
    :param var:
    :type var:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    mean_n_minus_one = mean
    var_n_minus_one = var
    n_minus_one = n
    n += 1
    mean = x_new / n + n_minus_one / n * mean_n_minus_one
    var = 1 / n_minus_one * ((x_new - mean) * (x_new - mean_n_minus_one) + (n - 2) * var_n_minus_one)
    return mean, var, n

@nb.njit
def sub_from_moments(x_new, mean, var, n):
    """
    Update existing mean and variance values by substracting a new data point.

    :param x_new:
    :type x_new:
    :param mean:
    :type mean:
    :param var:
    :type var:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    n_minus_one = n - 1
    mean_n = mean
    mean_n_minus_one = n / n_minus_one * (mean - x_new / n)
    var_n_minus_one = 1 / (n - 2) * (n_minus_one * var - (x_new - mean_n) * (x_new - mean_n_minus_one))
    n -= 1
    mean = mean_n_minus_one
    var = var_n_minus_one
    if var < 0:  # in case of numerical issues
        var = 1e-8
    return mean, var, n

@nb.njit
def find_peaks(array, lag, threshold, init_mean, init_var, fixed_var=0):
    """
    Find the peaks in an array and return an array with 1/0/-1.

    Based on https://stackoverflow.com/a/22640362/15216821.

    :param array: The array in which we want to detect peaks.
    :type array: list
    :param lag: The size of the window which we use for calculating a mean and variance.
    :type lag: int
    :param threshold: The sigma value of the threshold
    :type threshold: float
    :param init_mean: Here we can provide an initial mean value, otherwise this is extracted from the data.
    :type init_mean: float
    :param init_var: Here we can provide an initial variance value, otherwise this is extracted from the data.
    :type init_var: float
    :param fixed_var: In case you don't want to calculate the variance with a moving window, you can provide a fixed
        value.
    :type fixed_var: float
    :return: The array with 1/0/-1, corresponding to positive/negative peaks at the sample position. The means at all
        sample positions. The variances at all sample positions.
    :rtype: list
    """

    n = lag
    if init_mean is None:
        mean = np.mean(array[:lag])
    else:
        mean = init_mean
    if init_var is None:
        variance = var(array[:lag])
    else:
        variance = init_var

    signal = np.zeros(len(array))
    all_means = np.full(len(array), mean)
    all_vars = np.full(len(array), variance)

    for i in range(array.shape[0]):

        if fixed_var > 0:
            variance = fixed_var

        if np.abs(array[i] - mean) > threshold * np.sqrt(variance):
            if array[i] > mean:
                signal[i] = 1
            else:
                signal[i] = -1

        if i >= lag:
            mean, variance, n = add_to_moments(array[i], mean, variance, n)
            mean, variance, n = sub_from_moments(array[i - lag], mean, variance, n)
        all_means[i] = mean
        all_vars[i] = variance

    return signal, all_means, all_vars


@nb.njit
def get_triggers(array, lag, threshold, init_mean, init_var, look_ahead, fixed_var=0):
    """
    Trigger the peaks in an array and return their indices and properties.

    Based on https://stackoverflow.com/a/22640362/15216821.

    :param array: The array in which we want to detect peaks.
    :type array: list
    :param lag: The size of the window which we use for calculating a mean and variance.
    :type lag: int
    :param threshold: The sigma value of the threshold
    :type threshold: float
    :param init_mean: Here we can provide an initial mean value, otherwise this is extracted from the data.
    :type init_mean: float
    :param init_var: Here we can provide an initial variance value, otherwise this is extracted from the data.
    :type init_var: float
    :param look_ahead: Look this number of samples in the future if another peak is higher than this one and overwrite
        if so.
    :type look_ahead: int
    :param fixed_var: In case you don't want to calculate the variance with a moving window, you can provide a fixed
        value.
    :type fixed_var: float
    :return: The array with the triggered indices. The corresponding peak heights. The means at all
        sample positions. The variances at all sample positions.
    :rtype: list
    """

    n = lag
    if init_mean is None:
        mean = np.mean(array[:lag])
    else:
        mean = init_mean
    if init_var is None:
        variance = var(array[:lag])
    else:
        variance = init_var
    block = 0

    signal = []
    heights = []
    all_means = []
    all_vars = []

    for i in range(array.shape[0]):

        if fixed_var > 0:
            variance = fixed_var

        if array[i] - mean > threshold * np.sqrt(variance):
            if block == 0:
                height = np.max(array[i:i+look_ahead])
                idx = i + np.argmax(array[i:i+look_ahead])
                signal.append(idx)
                heights.append(height - 2 * np.sqrt(variance) - mean)
                all_means.append(mean)
                all_vars.append(variance)
                block += look_ahead
        if i >= lag:
            mean, variance, n = add_to_moments(array[i], mean, variance, n)
            mean, variance, n = sub_from_moments(array[i - lag], mean, variance, n)

        if block > 0:
            block -= 1

    return signal, heights, all_means, all_vars

@nb.njit
def var(x):
    """
    Numba surrogate for calculating the sample variance.

    :param x: The elements of which we want to calculate the variance.
    :type x: list
    :return: The variance of the array.
    :rtype: float
    """
    m = np.mean(x)
    return 1 / (x.shape[0] - 1) * np.sum((x - m) ** 2)
