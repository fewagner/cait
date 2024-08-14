from typing import Union, List
from functools import partial

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from tqdm.auto import tqdm

import cait.versatile as vai

from .triggerbase import trigger_base

####################################################
### FUNCTIONS IN THIS FILE HAVE NO TESTCASES YET ###
####################################################

def zscore_chunk(data: np.ndarray, record_length: int):
    """
    Calculates the moving z-score of 'data'.
    Found here: https://stackoverflow.com/a/47165379

    :param data: The data for which the z-score has to be calculated.
    :type data: np.ndarray
    :param record_length: The record length for the triggering (this determines the length of the running average/variance for the z-score).
    :type record_length: int

    :return: Moving z-score of data
    :rtype: np.ndarray
    """
    r = pd.Series(data).rolling(window=record_length)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)

    return np.array((data-m)/s)[record_length:]

def trigger_zscore(stream: ArrayLike,
                   record_length: int,
                   threshold: float = 5,
                   n_triggers: int = None,
                   chunk_size: int = 100,
                   apply_first: Union[callable, List[callable]] = None):
    """
    Trigger a single channel of a stream using a moving z-score. See :func:`cait.versatile.functions.trigger.triggerbase.trigger_base` for details on the implementation.

    :param stream: The stream channel to trigger.
    :type stream: ArrayLike
    :param threshold: The threshold (in z-scores) above which events should be triggered.
    :type threshold: float
    :param n_triggers: The number of events to trigger (might be more, depending on 'chunk_size'). E.g. useful to look at the first 100 triggered events. Defaults to None, i.e. all events in the stream are triggered
    :type n_triggers: int
    :param chunk_size: The number of record windows that are processed (i.e. z-score calculation + peak search) at a time.
    :type chunk_size: int
    :param apply_first: A function or list of functions to be applied to the stream data BEFORE the filter function is applied. E.g. ``lambda x: -x`` to trigger on the inverted stream.
    :type apply_first: Union[callable, List[callable]], optional

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]

    **Example:**

    .. code-block:: python
    
        import cait.versatile as vai

        # Construct stream object
        stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")
        # Perform triggering
        trigger_inds, amplitudes = vai.trigger_of(stream["ADC1"], 2**14)
        # Get trigger timestamps from trigger indices
        timestamps = stream.time[trigger_inds]
        # Plot trigger amplitude spectrum
        vai.Histogram(amplitudes)
    """
    # Before samples exceeding threshold are searched, the z-scores of the chunks are calculated
    filter_fnc = partial(zscore_chunk, record_length=record_length)

    # Trigger to get the trigger indices
    inds, _ =  trigger_base(stream=stream,
                            threshold=threshold,
                            filter_fnc=filter_fnc,
                            record_length=record_length,
                            n_triggers=n_triggers,
                            chunk_size=chunk_size,
                            apply_first=apply_first)
    
    if not inds: return [], []
    
    # The trigger_vals are given in z-scores.
    # Therefore, we also calculate a naïve pulse height after subtracting a constant baseline
    
    # Also apply the function for the pulse height calculation
    if apply_first is not None:
        if callable(apply_first): apply_first = [apply_first]
    else:
        apply_first = []

    # Record windows centered at 1/4*record_length
    before = int(record_length/4)
    after = int(3*record_length/4)

    # Slice to search peak in the interval (1/5, 2/5) of the record window
    sl = slice(int(record_length/5), int(2*record_length/5))

    phs = np.zeros(len(inds), dtype=np.float32)
    processing = apply_first + [vai.BoxCarSmoothing(), vai.RemoveBaseline()]

    for i, ind in enumerate(pbar := tqdm(inds, desc="Calculating pulse heights")):
        trace = stream[ind-before:ind+after]
        for p in processing: trace = p(trace)

        phs[i] = np.max(trace[sl])

    return inds, phs