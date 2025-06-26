from functools import partial
from typing import List, Union

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

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
    # The scipy function 'oaconvolve' does NOT assume the response function to be in wrap-around order.
    # Therefore, we have to shift it such that the maximum position is again at 1/4 of the record window
    offset = record_length // 4
    return sp.signal.oaconvolve(np.roll(np.fft.irfft(of), offset), data, mode="full")[
        record_length + offset : -2 * record_length + offset + 1
    ]


def filter_chunk_correlated(data: np.ndarray, of: np.ndarray, record_length: int):
    """
    Filters multi-channel 'data' with a correlated optimum filter 'of' according to the overlap-add-algorithm and sums them up to obtain a single, filtered trace, which is ready for triggering/maximum search.

    :param data: The data to filter.
    :type data: np.ndarray
    :param of: The filter to use.
    :type of: np.ndarray
    :param record_length: The record length as determined by the filter length (this determines, how many samples are discarded in the beginning and end of the data to avoid edge effects).
    :type record_length: int

    :return: Filtered chunk
    :rtype: np.ndarray
    """
    # The scipy function 'oaconvolve' does NOT assume the response function to be in wrap-around order.
    # Therefore, we have to shift it such that the maximum position is again at 1/4 of the record window
    offset = record_length // 4
    return np.sum(
        np.atleast_2d(
            sp.signal.oaconvolve(
                np.roll(np.fft.irfft(of), offset, axis=-1), data, mode="full"
            )
        ),
        axis=0,
    )[record_length + offset : -2 * record_length + offset + 1]


def trigger_of(
    stream: ArrayLike,
    threshold: float,
    of: np.ndarray,
    n_triggers: int = None,
    chunk_size: int = 100,
    apply_first: Union[callable, List[callable]] = None,
    n_processes: int = None,
):
    """
    Trigger a single channel of a stream using the optimum filter triggering algorithm described in https://edoc.ub.uni-muenchen.de/23762/. See :func:`cait.versatile.functions.trigger.triggerbase.trigger_base` for details on the implementation.

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
    :param n_processes: The number of processes to use to process chunks. If None, all available cores are utilized. Defaults to None
    :type n_processes: int, optional

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]

    **Example:**

    .. code-block:: python

        import cait.versatile as vai

        # Construct stream object
        stream = vai.Stream(hardware="vdaq2", src="path/to/stream_file.bin")
        # Get an optimum filter from somewhere (here, we get one from mock data but
        # this will not work for your stream data)
        of = vai.MockData().of

        # Perform triggering (use context to keep stream file opened)
        with stream:
            trigger_inds, amplitudes = vai.trigger_of(stream["ADC1"], 0.1, of)
        # Get trigger timestamps from trigger indices
        timestamps = stream.time[trigger_inds]
        # Plot trigger amplitude spectrum
        vai.Histogram(amplitudes)
    """
    # size of record window (as determined by the size of the filter)
    record_length = 2 * (of.shape[-1] - 1)

    # before samples exceeding threshold are searched, the chunks are filtered
    filter_fnc = partial(filter_chunk, of=of, record_length=record_length)

    return trigger_base(
        stream=stream,
        threshold=threshold,
        filter_fnc=filter_fnc,
        record_length=record_length,
        n_triggers=n_triggers,
        chunk_size=chunk_size,
        apply_first=apply_first,
        n_processes=n_processes,
    )


def trigger_ofc(
    streams: ArrayLike,
    threshold: float,
    ofc: np.ndarray,
    n_triggers: int = None,
    chunk_size: int = 100,
    apply_first: Union[callable, List[callable]] = None,
    n_processes: int = None,
):
    """
    Trigger multiple channels of a stream at the same time using a correlated (2d) optimum filter and the optimum filter triggering algorithm described in https://edoc.ub.uni-muenchen.de/23762/. See :func:`cait.versatile.functions.trigger.triggerbase.trigger_base` for details on the implementation. The correlated filtering returns A SINGLE voltage trace (even for multiple input channels), hence, only a combined trigger threshold is required.

    Apart from the fact that multi-channel 'streams' and 'ofc' are required here, the usage is identical to :func:`cait.versatile.trigger_of`.

    :param streams: The stream channels to trigger.
    :type streams: ArrayLike
    :param threshold: The threshold (in Volts) above which events should be triggered.
    :type threshold: float
    :param ofc: The correlated (2d) optimum filter to be used for filtering (it is assumed that the filter's first entry is set to zero to correctly remove the offset).
    :type ofc: np.ndarray
    :param n_triggers: The number of events to trigger (might be more, depending on 'chunk_size'). E.g. useful to look at the first 100 triggered events. Defaults to None, i.e. all events in the stream are triggered
    :type n_triggers: int
    :param chunk_size: The number of record windows that are processed (i.e. filter + peak search) at a time.
    :type chunk_size: int
    :param apply_first: A function or list of functions to be applied to the stream data BEFORE the filter function is applied. E.g. ``lambda x: -x`` to trigger on the inverted stream.
    :type apply_first: Union[callable, List[callable]], optional
    :param n_processes: The number of processes to use to process chunks. If None, all available cores are utilized. Defaults to None
    :type n_processes: int, optional

    :return: Tuple of trigger indices and trigger heights.
    :rtype: Tuple[List[int], List[float]]
    """
    # size of record window (as determined by the size of the filter)
    record_length = 2 * (ofc.shape[-1] - 1)

    # before samples exceeding threshold are searched, the chunks are filtered
    filter_fnc = partial(filter_chunk_correlated, of=ofc, record_length=record_length)

    return trigger_base(
        stream=streams,
        threshold=threshold,
        filter_fnc=filter_fnc,
        record_length=record_length,
        n_triggers=n_triggers,
        chunk_size=chunk_size,
        apply_first=apply_first,
        n_processes=n_processes,
    )
