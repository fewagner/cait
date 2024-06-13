from typing import Union, List

import numpy as np
from numpy.typing import ArrayLike
import numba as nb

from tqdm.auto import tqdm

####################################################
### FUNCTIONS IN THIS FILE HAVE NO TESTCASES YET ###
####################################################

@nb.njit
def search_chunk(data: np.ndarray, threshold: float, record_length: int, skip_first: int = 0):
    """
    Searches samples in 'data' exceeding the 'threshold' according to the algorithm discussed in https://edoc.ub.uni-muenchen.de/23762/.

    :param data: The data to search.
    :type data: np.ndarray
    :param threshold: The threshold (in Volts) above which events should be triggered.
    :type threshold: float
    :param record_length: The record length as determined by the filter (this determines, how many samples are not searched in the end of the data)
    :type record_length: int
    :param skip_first: The number of samples to skip in the beginning of the chunk (needed if beginning of chunk should be blinded)
    :type skip_first: int, optional

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """
    trigger_inds = []
    trigger_vals = []
    
    search_len = len(data)-record_length

    inds = iter(range(skip_first, search_len))
    for i in inds:
        if data[i] > threshold:
            j = np.argmax(data[i:i+record_length])
            trigger_inds.append(i+j)
            trigger_vals.append(data[i+j])
            
            if i+j+record_length//2 > search_len: 
                break
            else:
                for _ in range(j+record_length//2): 
                    next(inds)

    return trigger_inds, trigger_vals

def trigger_base(stream: ArrayLike,
                 threshold: float,
                 filter_fnc: callable,
                 record_length: int,
                 n_triggers: int = None,
                 chunk_size: int = 100,
                 apply_first: Union[callable, List[callable]] = None):
    """
    Trigger a single channel of a stream after pre-processing the stream. This function is used both for optimum filter triggering as well as for z-score triggering.

    :param stream: The stream channel to trigger.
    :type stream: ArrayLike
    :param threshold: The threshold above which events should be triggered (interpretation depends on 'filter_fnc').
    :type threshold: float
    :param filter_fnc: The function to be applied to the data before triggering.
    :type filter_fnc: callable
    :param record_length: The desired record length (this determines the dead-time after a trigger)
    :type record_length: int
    :param n_triggers: The number of events to trigger (might be more, depending on 'chunk_size'). E.g. useful to look at the first 100 triggered events. Defaults to None, i.e. all events in the stream are triggered
    :type n_triggers: int
    :param chunk_size: The number of record windows that are processed (i.e. filter + peak search) at a time.
    :type chunk_size: int
    :param apply_first: A function or list of functions to be applied to the stream data BEFORE the filter function is applied. E.g. ``lambda x: -x`` to trigger on the inverted stream.
    :type apply_first: Union[callable, List[callable]], optional

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """

    if apply_first is not None:
        if type(apply_first) in [list, tuple]:
            if not all([callable(f) for f in apply_first]):
                raise TypeError("All entries of list 'apply_first' must be callable.")
        elif callable(apply_first):
            apply_first = [apply_first]
        else:
            raise TypeError(f"Unsupported type '{type(apply_first)}' for input argument 'apply_first'.")
    else:
        apply_first = []

    # total number of samples in the stream
    stream_length = len(stream)
    # number of samples to be searched for triggers at a time
    search_length = chunk_size*record_length

    # number of such search chunks (the first and last record window is not
    # searched because those regions cannot be filtered correctly)
    n_search_areas = (stream_length - 3*record_length)//search_length
    remainder = (stream_length - 3*record_length)%search_length

    search_area_sizes = [search_length]*n_search_areas
    if remainder!=0: search_area_sizes += [remainder]

    # create lists of start and end indices of search areas
    starts = [record_length + x*search_length for x in range(len(search_area_sizes))]
    ends = [s+sz for s,sz in zip(starts, search_area_sizes)]

    # initialize a chunk for the filtering (this has to start one record window
    # early and end two record windows after the search stop)
    chunk = np.zeros(search_length + 3*record_length)

    # Initialize the lists that will collect the triggers
    trigger_inds = []
    trigger_vals = []
    triggers_found = 0

    # Number of samples to skip in the beginning of a chunk
    # (needed if beginning of chunk needs to be blinded 
    # because trigger was found close to edge of previous chunk)
    skip_first = 0

    for s, e, sz in zip(pbar := tqdm(starts), ends, search_area_sizes):
        chunk[:sz+3*record_length] = stream[s-record_length:e+2*record_length]

        for f in apply_first: chunk[:] = f(chunk)

        filtered_chunk = filter_fnc(chunk)
        inds, vals = search_chunk(filtered_chunk, threshold, record_length, skip_first=skip_first)

        trigger_inds += [s+i for i in inds]
        trigger_vals += vals
        triggers_found += len(inds)

        pbar.set_postfix({"triggers found": triggers_found})
        if (n_triggers is not None) and (triggers_found > n_triggers): break

        chunk[:] = 0

        # If trigger is found in last window of search area, we blind the
        # beginning of the following chunk
        if inds and (s + inds[-1] > e):
            skip_first = s + inds[-1] - e + record_length//2
        else:
            skip_first = 0
        
    return trigger_inds, trigger_vals