import numpy as np
import scipy as sp
import numba as nb

from tqdm.auto import tqdm

def filter_chunk(data, of, record_length):
    return sp.signal.oaconvolve(data, np.fft.irfft(of))[record_length:-record_length]

@nb.njit
def search_chunk(data, threshold, record_length):
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
        #chunk[:sz+3*record_length] = stream[s-record_length:e+2*record_length]

        filterd_chunk = filter_chunk(chunk, of, record_length)
        inds, vals = search_chunk(filterd_chunk, threshold, record_length)

        trigger_inds += [s+i for i in inds]
        trigger_vals += vals
        triggers_found += len(inds)

        pbar.set_postfix({"triggers found": triggers_found})
        if (n_triggers is not None) and (triggers_found > n_triggers): break

        chunk[:] = 0
        
    return trigger_inds, trigger_vals