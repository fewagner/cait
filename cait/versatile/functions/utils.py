import numpy as np
from typing import List, Tuple

# Has no testcase yet
# also implement a timestamp_offset of some sorts
def timestamps_to_timedict(timestamps_us: np.ndarray,
                           hours_offset: float = 0):
    """
    Convert microsecond timestamps (as obtained, e.g., from a steam or an .rdt file) to `hours`, `time_s` and `time_mus` (as used by all `cait` routines).
     
    :param timestamps_us: Array of timestamps in seconds
    :type timestamps_us: np.ndarray
    :param hours_offset: Possible offset of the 'hours' data. If set to 0, the first timestamp in `timestamps_us` marks 0 hours. Defaults to 0.
    :type hours_offset: float, optional
    
    :return: Dictionary with keys `hours`, `time_s` and `time_mus`.
    :rtype: dict

    >>> # If we have 'timestamps' for some events, we can include the corresponding
    >>> # 'hours', 'time_s' and 'time_mus' datasets in a DataHandler instance 'dh' as:
    >>> time_data = timestamps_to_timedict(timestamps)
    >>> dh.set(group="events", **time_data)
    """
    hours = (timestamps_us - np.min(timestamps_us))/1e6/3600 + hours_offset # should be float64
    time_s = np.array(timestamps_us/1e6, dtype=np.int32) # should be int64
    time_mus = np.array(timestamps_us - time_s*int(1e6), dtype=np.int32) # should be int64

    return dict(hours=hours, time_s=time_s, time_mus=time_mus)

# has no test-case (yet)
# also implement a timestamp_offset of some sorts
def timestamps_to_hours(timestamps_us: np.ndarray, 
                        timestamps_s: np.ndarray = None, 
                        hours_offset: float = 0):
    """
    Converts a microseconds timestamps array to an hours array which starts at `hours_offset`. If microsecond timestamps (`timestamps_us`) AND seconds timestamps (`timestamps_s`) are provided, the microsecond timestamps are interpreted as starting from the respective seconds timestamps, i.e. `timestamps_to_hours(ts_us, ts_s)` is equivalent to `timestamps_to_hours(int(1e6)*ts_s+ts_us)`.

    :param timestamps_us: Array of timestamps in microseconds
    :type timestamps_us: np.ndarray
    :param timestamps_s: Array of timestamps in seconds. If provided, the microsecond timestamps are interpreted as starting from the respective seconds timestamps. Defaults to None.
    :type timestamps_s: np.ndarray, optional
    :param hours_offset: Possible offset of the hours data. If set to 0, the first timestamp marks 0 hours. Defaults to 0.
    :type hours_offset: float, optional

    :return: Array of hours
    :rtype: np.ndarray
    """
    if isinstance(timestamps_us, list): timestamps_us = np.array(timestamps_us)
    if isinstance(timestamps_s, list): timestamps_s = np.array(timestamps_s)
    if timestamps_s is not None:
        timestamps_us = int(1e6)*np.array(timestamps_s, dtype=np.int64) + np.array(timestamps_us, dtype=np.int64)
    
    return (timestamps_us - np.min(timestamps_us))/1e6/3600 + hours_offset

def timestamp_coincidence(a: List[int], b: List[int], interval: Tuple[int]):
    """
    Determine the coincidence of timestamps in two separate arrays.

    Constructs half-open intervals `[a+interval[0], a+interval[1])` around timestamps in **a** and checks which timestamps in **b** fall into these intervals.

    Note that the timestamps have to be strictly monotonically increasing! However, it is possible that the resulting intervals overlap. In that case, the intervals are symmetrically shrunk to create two smaller intervals which touch in the middle. E.g. if the intervals are [12, 20) and [15, 23), the resulting new intervals will be [12, 18) and [18, 23).

    :param a: Array of timestamps in microseconds
    :type a: List[int]
    :param b: Array of timestamps in microseconds
    :type b: List[int]
    :param interval: Tuple of length 2 which specifies how many microseconds before and after a timestamp should be considered to be in coincidence. E.g. (-10,20) means that for the timestamp 100, the interval [90,120) would be considered. Note that also intervals (1,10) and (-10,-5) are valid.
    :type interval: Tuple[int]

    :return:

                - INDICES of **b** in coincidence with **a**, 
                - corresponding INDICES of **a** which elements of **b** are in coincidence with,
                - INDICES of **b** NOT in coincidence with **a**

    :rtype: Tuple

    **Example:**

    .. code-block:: python

        a = np.array([10, 20, 30, 40])
        b = np.array([1, 11, 35, 42, 45])
        inside, coincidence_inds, outside = vai.timestamp_coincidence(a,b,(-1,3))

        inside # array([1, 3])
        outside # array([0, 2, 4])
        coincidence_inds # array([0, 3])
        b[inside] # array([11, 42]) = corresponding timestamps
        b[outside] # array([ 1, 35, 45]) = corresponding timestamps
        a[coincidence_ind] # array([10, 40])
    """
    # Construct bin edges: make array where first row are lower bin edges (a-interval[0]) 
    # and second row are upper bin edges (a+interval[1]). Afterwards the array is flattened
    # in 'F' order, i.e. column-major.
    # Example: for a = [10, 20, 30] and interval = (1, 2), edges = [9, 12, 19, 22, 29, 32]
    edges = np.array([np.array(a)+interval[0], np.array(a)+interval[1]]).flatten(order='F')
    
    # if bins overlap, we shrink them symmetrically
    overlap_inds = np.where(np.diff(edges)<0)[0]
    
    overlap_sizes = edges[overlap_inds] - edges[overlap_inds+1]
    
    edges[overlap_inds] = edges[overlap_inds] - overlap_sizes//2
    edges[overlap_inds+1] = edges[overlap_inds+1] + overlap_sizes//2 + 1

    # Do binning (right argument specifies the half-open interval)
    bin_inds = np.digitize(np.array(b), edges, right=False)

    # All odd-numbered bin-indices are coincident, all even-numbered ones are not
    mask_even = np.mod(bin_inds, 2) == 0

    # Helper array to get indices of array b instead of the timestamps themselves
    inds = np.arange(len(b))

    # Also return the indices of a which the elements of b are in coincidence with.
    inds_a = np.array((bin_inds[~mask_even]-1)/2, dtype=int)

    return (inds[~mask_even], inds_a, inds[mask_even])

def sample_noise(trigger_inds: List[int], record_length: int, n_samples: int = None):
    """
    Get stream indices of noise traces. Record windows of length `record_length` are aligned around `trigger_inds` (at 1/4th of the record length) and noise indices are only sampled from large enough intervals *outside* these windows (i.e. only if at least one noise window of length `record_length` fits within such gaps). Note that the selected noise traces can still contain pulses and artifacts.

    :param trigger_inds: The indices (*not* timestamps) of the triggered events. To get as clean noise traces as possible, this should include *all* triggers, i.e. also testpulses for example.
    :type trigger_inds: List[int]
    :param record_length: The length of the record window in samples. This has to match the record length which was used for obtaining `trigger_inds` for meaningful results.
    :type record_length: int
    :param n_samples: If specified, a total number of at most `n_samples` are returned. The actual number of returned noise traces can be lower depending on the available 'empty space' between actual triggers.
    :type n_samples: int, optional

    :return: List of indices for noise traces. The record windows can be recovered by aligning them at ``int(record_length/4)`` around ``indices``.
    :rtype: List[int]
    """

    if n_samples is None: n_samples = np.inf

    # Index of trigger relative to beginning of record window
    onset = int(record_length/4)

    noise_inds = []
    all_inds = trigger_inds.copy()

    # In case n_samples is not set, the loop runs until all intervals are exhausted, i.e. until no more vacancies are found.
    while len(noise_inds) < n_samples:
        # It will be easier below if we sort trigger_inds
        all_inds.sort()

        # Intervals around triggers (each column corresponds to one interval)
        # By interval we mean interval[0] is the first and interval[1] the last index of the record window
        intervals = np.array([np.array(all_inds)-onset, 
                              np.array(all_inds)+record_length-onset-1])
        
        # Find large enough gaps. Those gaps start by interval_i[1]+1 and end at interval_(i+1)[0]-1. The number of elements in these gaps is then one larger than the difference of the boundaries.
        gap_sizes = (intervals[0,1:]-1) - (intervals[1,:-1]+1) + 1
        mask = gap_sizes >= record_length
        n_vacancies = np.sum(mask)

        if n_vacancies < 1: break

        # Construct 'vacant' intervals from 'occupied' ones
        vacant_intervals = np.array([intervals[1,:-1][mask] + 1, 
                                     intervals[0,1: ][mask] - 1])
    
        # Since the triggers are aligned WITHIN the record window, the range for possible noise indices is narrower
        vacant_intervals[0] += onset
        vacant_intervals[1] += onset - record_length + 1

        # Draw random index in available intervals (np.random.randint uses half-open intervals [low,high) which is why we add +1)
        draws = np.random.randint(low=vacant_intervals[0], high=vacant_intervals[1]+1)

        noise_inds.extend(list(draws))
        all_inds.extend(list(draws))

    # Since we draw all random numbers at once, it can happen that we end up with more samples than required. Nevertheless, it's easier to draw them all at once.
    # To remain unbiased, the required n samples are drawn randomly (because `noise_inds` is currently sorted in ascending order)
    if len(noise_inds) > n_samples: 
         # Draw without replacement. According to stackexchange, default_rng().choice() is the fastest option
         noise_inds = np.random.default_rng().choice(noise_inds, n_samples, replace=False)
    
    return noise_inds