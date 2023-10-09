import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

# has no test-case (yet)
# Better implementation: timestamps_us only also possible
def timestamps_to_hours(timestamp_s: List[int], 
                        microseconds: List[int] = None, 
                        start_timestamp_s: int = None, 
                        start_microseconds: int = None):
    """
    Converts timestamps and microseconds to an hours array which starts at start_timestamp. If no start_timestamp is specified, the earliest timestamp_s is used. 

    :param timestamp_s: Array of timestamps in seconds
    :type timestamp_s: List[int]
    :param microseconds: Possible microsecond corrections to the second timestamps (i.e. actual timestamps in microseconds would be timestamp_s*1e6 + microseconds)
    :type microseconds: List[int], Default: None (no microsecond corrections are used)
    :param start_timestamp: The seconds timestamp that is used as start of the array. If None is specified, the earliest timestamp_s is used. 
    :type start_timestamp: int
    :param start_microseconds: The microseconds correction that is used as start of the array. If None is specified, the microseconds correction corresponding to the earliest timestamp_s is used. 
    :type start_microseconds: int

    :return: Array of hours
    :rtype: `~class:numpy.ndarray`
    """

    if np.logical_xor(start_timestamp_s==None, start_microseconds==None):
            print("You have to specify 'start_timestamp_s' and 'start_microseconds_s' together")
            return
    
    if isinstance(timestamp_s, list): timestamp_s = np.array(timestamp_s)
    
    if microseconds is None:  
        microseconds = np.zeros_like(timestamp_s, dtype='int32')
    elif isinstance(microseconds, list): 
        microseconds = np.array(microseconds, dtype='int32')

    if start_timestamp_s is None:
        earliest_ind = np.argmin(timestamp_s)
        start_timestamp_s = timestamp_s[earliest_ind]
        start_microseconds = microseconds[earliest_ind]    

    datetimes = [datetime.fromtimestamp(s, tz=timezone.utc) + timedelta(microseconds=int(mus)) 
                 for s, mus in zip(timestamp_s, microseconds)]
    datetime_start = datetime.fromtimestamp(start_timestamp_s, tz=timezone.utc) + timedelta(microseconds=int(start_microseconds))
    durations = [dt - datetime_start for dt in datetimes]

    return np.array([dur.total_seconds() for dur in durations])/3600

def timestamp_coincidence(a: List[int], b: List[int], interval: Tuple[int]):
    """
    Determine the coincidence of timestamps in two separate arrays.

    Constructs half-open intervals `[a+interval[0], a+interval[1])` around timestamps in `a` and checks which timestamps in `b` fall into these intervals.
    Returns tuple 
    (
     INDICES of `b` in coincidence with `a`, 
     corresponding INDICES of `a` which elements of `b` are in coincidence with,
     INDICES of `b` NOT in coincidence with `a`,
    )

    Note that the intervals have to be non-overlapping to assure non-ambiguous binning!

    :param a: Array of timestamps in microseconds
    :type a: List[int]
    :param b: Array of timestamps in microseconds
    :type b: List[int]
    :param interval: Tuple of length 2 which specifies how many microseconds before and after a timestamp should be considered to be in coincidence. E.g. (-10,20) means that for the timestamp 100, the interval [90,120) would be considered. Note that also intervals (1,10) and (-10,-5) are valid.
    :type interval: Tuple[int]

    :return: (
              INDICES of `b` in coincidence with `a`, 
              corresponding INDICES of `a` which elements of `b` are in coincidence with,
              INDICES of `b` NOT in coincidence with `a`,
              )
    :rtype: Tuple

    >>> a = np.array([10, 20, 30, 40])
    >>> b = np.array([1, 11, 35, 42, 45])
    >>> inside, coincidence_inds, outside = vai.utils.timestamp_coincidence(a,b,(-1,3))

    >>> ind_in # array([1, 3])
    >>> ind_out # array([0, 2, 4])
    >>> ind_coin # array([0, 3])
    >>> b[inside] # array([11, 42]) = corresponding timestamps
    >>> b[outside] # array([ 1, 35, 45]) = corresponding timestamps
    >>> a[coincidence_ind] # array([10, 40])
    """
    # Construct bin edges: make array where first row are lower bin edges (a-interval[0]) 
    # and second row are upper bin edges (a+interval[1]). Afterwards the array is flattened
    # in 'F' order, i.e. column-major.
    # Example: for a = [10, 20, 30] and interval = (1, 2), edges = [9, 12, 19, 22, 29, 32]
    edges = np.array([np.array(a)+interval[0], np.array(a)+interval[1]]).flatten(order='F')

    # Do binning (right argument specifies the half-open interval)
    bin_inds = np.digitize(np.array(b), edges, right=False)

    # All odd-numbered bin-indices are coincident, all even-numbered ones are not
    mask_even = np.mod(bin_inds, 2) == 0

    # Helper array to get indices of array b instead of the timestamps themselves
    inds = np.arange(len(b))

    # Also return the indices of a which the elements of b are in coincidence with.
    inds_a = np.array((bin_inds[~mask_even]-1)/2, dtype=int)

    return (inds[~mask_even], inds_a, inds[mask_even])

# HAS NO TEST CASE YET
def sample_noise(trigger_inds: List[int], record_length: int, alignment: float = 1/4, n_samples: int = None):
    """
    Get stream indices of noise traces. Record windows of length `record_length` are aligned around `trigger_inds` (using `alignment`) and noise indices are only sampled from large enough intervals *outside* these windows (i.e. only if at least one noise window of length `record_length` fits within such gaps). Note that the selected noise traces can still contain pulses and artifacts.

    :param trigger_inds: The indices (*not* timestamps) of the triggered events. To get as clean noise traces as possible, this should include *all* triggers, i.e. also testpulses for example.
    :type trigger_inds: List[int]
    :param record_length: The length of the record window in samples. This has to match the record length which was used for obtaining `trigger_inds` for meaningful results.
    :type record_length: int
    :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index. Defaults to 1/4.
    :type alignment: float, optional
    :param n_samples: If specified, a total number of at most `n_samples` are returned. The actual number of returned noise traces can be lower depending on the available 'empty space' between actual triggers.
    :type n_samples: int, optional

    :return: List of indices for noise traces. The record windows can be recovered using `record_length` and `alignment`.
    :rtype: List[int]
    """
    if 0 > alignment or 1 < alignment:
            raise ValueError("'alignment' has to be in the interval [0,1]")

    if n_samples is None: n_samples = np.inf

    # Index of trigger relative to beginning of record window
    onset = int(alignment*record_length)

    noise_inds = []

    # In case n_samples is not set, the loop runs until all intervals are exhausted, i.e. until no more vacancies are found.
    while len(noise_inds) < n_samples:
        # It will be easier below if we sort trigger_inds
        trigger_inds.sort()

        # Intervals around triggers (each column corresponds to one interval)
        # By interval we mean interval[0] is the first and interval[1] the last index of the record window
        intervals = np.array([np.array(trigger_inds)-onset, 
                              np.array(trigger_inds)+record_length-onset-1])
        
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
        trigger_inds.extend(list(draws))

    # Since we draw all random numbers at once, it can happen that we end up with more samples than required. Nevertheless, it's easier to draw them all at once.
    # To remain unbiased, the required n samples are drawn randomly (because `noise_inds` is currently sorted in ascending order)
    if len(noise_inds) > n_samples: 
         # Draw without replacement. According to stackexchange, default_rng().choice() is the fastest option
         noise_inds = np.random.default_rng().choice(noise_inds, n_samples, replace=False)
    
    return noise_inds