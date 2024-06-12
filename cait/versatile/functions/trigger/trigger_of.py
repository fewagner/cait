from typing import Union, List
from functools import partial

import numpy as np
from numpy.typing import ArrayLike
import scipy as sp

from .triggerbase import trigger_base

####################################################
### FUNCTIONS IN THIS FILE HAVE NO TESTCASES YET ###
####################################################

def filter_chunk(data: np.ndarray, of: np.ndarray, record_length: int):
    """
    Filters 'data' with an optimum filter 'of' according to the overlap-add-algorithm.

    :param data: The data to filter.
    :type data: np.ndarray
    :param of: The filter to use.
    :type of: np.ndarray
    :param record_length: The record length as determined by the filter length (this determines, how many samples are discarded in the beginning and end of the data to avoid edge effects).
    :type record_length: int

    :return: Filtered chunk
    :rtype: np.ndarray
    """
    return sp.signal.oaconvolve(data, np.fft.irfft(of))[record_length:-record_length]

def trigger_of(stream: ArrayLike,
               threshold: float, 
               of: np.ndarray, 
               n_triggers: int = None,
               chunk_size: int = 100,
               apply_first: Union[callable, List[callable]] = None):
    """
    Trigger a single channel of a stream using the optimum filter triggering algorithm described in https://edoc.ub.uni-muenchen.de/23762/. 

    :param stream: The stream channel to trigger.
    :type stream: ArrayLike
    :param threshold: The threshold (in Volts) above which events should be triggered.
    :type threshold: float
    :param of: The optimum filter to be used for filtering (it is assumed that the filter's first entry is set to zero to correctly remove the offset).
    :type of: np.ndarray
    :param n_triggers: The number of events to trigger (might be more, depending on 'chunk_size'). E.g. useful to look at the first 100 triggered events. Defaults to None, i.e. all events in the stream are triggered
    :type n_triggers: int
    :param chunk_size: The number of record windows that are processed (i.e. filter + peak search) at a time.
    :type chunk_size: int
    :param apply_first: A function or list of functions to be applied to the stream data BEFORE the filter function is applied. E.g. ``lambda x: -x`` to trigger on the inverted stream.
    :type apply_first: Union[callable, List[callable]], optional

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]

    **Example:**
    ::
        import cait.versatile as vai

        # Construct stream object
        stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")
        # Get an optimum filter from somewhere (here, we get one from mock data but
        # this will not work for your stream data)
        of = vai.MockData().of
        
        # Perform triggering
        trigger_inds, amplitudes = vai.trigger_of(stream["ADC1"], 0.1, of)
        # Get trigger timestamps from trigger indices
        timestamps = stream.time[trigger_inds]
        # Plot trigger amplitude spectrum
        vai.Histogram(amplitudes)
    """
    # size of record window (as determined by the size of the filter)
    record_length = 2*(of.shape[-1] - 1)

    # before samples exceeding threshold are searched, the chunks are filtered
    filter_fnc = partial(filter_chunk, of=of, record_length=record_length)

    return trigger_base(stream=stream,
                        threshold=threshold,
                        filter_fnc=filter_fnc,
                        record_length=record_length,
                        n_triggers=n_triggers,
                        chunk_size=chunk_size,
                        apply_first=apply_first)