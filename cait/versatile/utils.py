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

# Will have to change in some way (maybe output timestamps, too)
def timestamp_coincidence(a: List[int], b: List[int], interval: Tuple[int]):
    """
    Determine the coincidence of timestamps in two separate arrays.

    Constructs half-open intervals `[a-interval[0], a+interval[1])` around timestamps in `a` and checks which timestamps in `b` fall into these intervals.
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
    :param interval: Tuple of length 2 which specifies how many microseconds before and after a timestamp should be considered to be in coincidence. E.g. (10,20) means that for the timestamp 100, the interval [90,120) would be considered.
    :type interval: Tuple[int]

    :return: (
              INDICES of `b` in coincidence with `a`, 
              corresponding INDICES of `a` which elements of `b` are in coincidence with,
              INDICES of `b` NOT in coincidence with `a`,
              )
    :rtype: Tuple

    >>> a = np.array([10, 20, 30, 40])
    >>> b = np.array([1, 11, 35, 42, 45])
    >>> inside, coincidence_inds, outside = vai.utils.timestamp_coincidence(a,b,(1,3))

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
    edges = np.array([np.array(a)-interval[0], np.array(a)+interval[1]]).flatten(order='F')

    # Do binning (right argument specifies the half-open interval)
    bin_inds = np.digitize(np.array(b), edges, right=False)

    # All odd-numbered bin-indices are coincident, all even-numbered ones are not
    mask_even = np.mod(bin_inds, 2) == 0

    # Helper array to get indices of array b instead of the timestamps themselves
    inds = np.arange(len(b))

    # Also return the indices of a which the elements of b are in coincidence with.
    inds_a = np.array((bin_inds[~mask_even]-1)/2, dtype=int)

    return (inds[~mask_even], inds_a, inds[mask_even])