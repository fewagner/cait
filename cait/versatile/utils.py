import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List

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