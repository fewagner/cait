import numpy as np
import scipy as sp
import numba as nb

from tqdm.auto import tqdm

def filter_chunk(data: np.ndarray, of: np.ndarray, record_length: int):
    """
    Filters 'data' with an optimum filter 'of' according to the overlap-add-algorithm.

    :param data: The data filter.
    :type data: np.ndarray
    :param of: The filter to use.
    :type of: np.ndarray
    :param record_length: The record length as determined by the filter length (this determines, how many samples are discarded in the beginning and end of the data to avoid edge effects).
    :type record_length: int

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """
    return sp.signal.oaconvolve(data, np.fft.irfft(of))[record_length:-record_length]

@nb.njit
def search_chunk(data: np.ndarray, threshold: float, record_length: int):
    """
    Searches samples in 'data' exceeding the 'threshold' according to the algorithm discussed in https://edoc.ub.uni-muenchen.de/23762/.

    :param data: The data to search.
    :type data: np.ndarray
    :param threshold: The threshold (in Volts) above which events should be triggered.
    :type threshold: float
    :param record_length: The record length as determined by the filter (this determines, how many samples are not searched in the end of the data)
    :type record_length: int

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """
    trigger_inds = []
    trigger_vals = []
    
    search_len = len(data)-record_length

    inds = iter(range(search_len))
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

def trigger_of(stream, 
               key: str, 
               threshold: float, 
               of: np.ndarray, 
               n_triggers: int = None,
               chunk_size: int = 100):
    """
    Trigger a single channel of a stream object using the optimum filter triggering algorithm described in https://edoc.ub.uni-muenchen.de/23762/. 

    :param stream: The stream object with the channel to trigger.
    :type stream: StreamBaseClass
    :param key: The name of the channel in 'stream' to trigger.
    :type key: str
    :param threshold: The threshold (in Volts) above which events should be triggered.
    :type threshold: float
    :param of: The optimum filter to be used for filtering (it is assumed that the filter's first entry is set to zero to correctly remove the offset).
    :type of: np.ndarray
    :param n_triggers: The number of events to trigger (might be more, depending on 'chunk_size'). E.g. useful to look at the first 100 triggered events. Defaults to None, i.e. all events in the stream are triggered
    :type n_triggers: int
    :param chunk_size: The number of record windows that are processed (i.e. filter + peak search) at a time.
    :type chunk_size: int

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """
    # size of record window (as determined by the size of the filter)
    record_length = 2*(of.shape[-1] - 1)
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

    for s, e, sz in zip(pbar := tqdm(starts), ends, search_area_sizes):
        chunk[:sz+3*record_length] = stream[key, s-record_length:e+2*record_length, "as_voltage"]

        filtered_chunk = filter_chunk(chunk, of, record_length)
        inds, vals = search_chunk(filtered_chunk, threshold, record_length)

        trigger_inds += [s+i for i in inds]
        trigger_vals += vals
        triggers_found += len(inds)

        pbar.set_postfix({"triggers found": triggers_found})
        if (n_triggers is not None) and (triggers_found > n_triggers): break

        chunk[:] = 0
        
    return trigger_inds, trigger_vals