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
    
    search_len = len(data) - record_length

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
    Trigger a single channel of a stream after pre-processing the stream. This function is used both for optimum filter triggering as well as for z-score triggering. See below for a description on how the algorithm works.

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

    **Description of the Trigger Algorithm:**

    The stream is split into (not necessarily equally sized) chunks which are processed at once. This reduces file access and larger chunks are generally preferred if sufficient memory is available. To correctly filter a chunk (optimum filtering or moving z-score), an additional record window *before* the chunk is needed (because the filter length is one record length). For this reason, the very first record window of the stream is discarded, i.e. not searched for triggers. Additionally, one record window *after* the chunk ends is required to not miss any edge cases as marked with numbers 1-3 in the figure (see later). We call the start of a chunk :math:`s`, the end :math:`e` and the record length :math:`N`. The triggering now proceeds as follows:

    1. A chunk is selected and the part of the stream from :math:`s-N` to :math:`e+N` is continuously filtered. The first record window of the filter output is discarded. The valid samples (:math:`s` to :math:`e+N`) to be searched for triggers are shown in the figure.
    2. The cursor is placed on the first sample, :math:`s`, and progresses through the chunk until a sample above threshold is found.
    3. If a sample exceeding the threshold is found, the maximum position of the :math:`N` samples *after* that sample is considered the trigger sample :math:`j`. The cursor is moved to :math:`j+N/2`, i.e. the trigger is blinded for half a record window. The process finishes if :math:`j+N/2` falls outside the chunk (i.e. :math:`j+N/2>e`), or if the cursor reaches the end of the chunk, :math:`e`.
    4. After finishing a chunk, the next one is loaded. We have to be careful not to double count triggers, though: If the last chunk had a trigger later than :math:`e-N/2`, the blinding process affects the following chunk. This is visualised by triggers 1-3 in the figure: 1 is fine, because more than half a record window remains in the chunk and blinding in the following chunk is not required. 2 and 3 on the other hand require blinding of the first :math:`j+N/2-e` samples of the following chunk.

    .. image:: media/TriggerAlgorithm.png

    **Nota bene:** 

    - One could be lead to believe that the additional record window *after* the end of the chunk is unnecessary, but this would introduce a subtle issue in the triggering process: If we find a sample above threshold at position 1 or 2, we cannot search the following :math:`N` samples for a maximum, and if we just stopped the search :math:`N` samples before the end of the chunk, we could miss triggers because those samples are not searched in the subsequent chunk either. Therefore, we have to search until we reach :math:`e`. If we find a sample above threshold just before (or at) :math:`e`, we still have enough samples left to correctly determine the maximum of the upcoming :math:`N` samples. 
    
    - In the end of the stream we have to discard 2 (!) record windows even though one appears to be sufficient at first glance. As discussed in the previous bullet point, the algorithm described above could lead to a trigger index as late as :math:`e+N`. To leave enough samples to read the voltage trace of the triggered event in the analysis, an additional record window is kept (even though in the common convention - that the trigger is placed at :math:`N/4` in the record window - :math:`3N/4` samples would technically be sufficient).
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
    # chunk = np.zeros(search_length + 3*record_length)
    chunk = np.zeros(search_length + 2*record_length)

    # Initialize the lists that will collect the triggers
    trigger_inds = []
    trigger_vals = []
    triggers_found = 0

    # Number of samples to skip in the beginning of a chunk
    # (needed if beginning of chunk needs to be blinded 
    # because trigger was found close to edge of previous chunk)
    skip_first = 0

    for s, e, sz in zip(pbar := tqdm(starts), ends, search_area_sizes):
        # chunk[:sz+3*record_length] = stream[s-record_length:e+2*record_length]
        chunk[:sz+2*record_length] = stream[s-record_length:e+record_length]

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
        # if inds and (s + inds[-1] > e):
        if inds and (s + inds[-1] > e-record_length//2):
            # skip_first = s + inds[-1] - e + record_length//2
            skip_first = s + inds[-1] + record_length//2 - e
        else:
            skip_first = 0
        
    return trigger_inds, trigger_vals