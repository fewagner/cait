from functools import partial
from typing import List, Union

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

from .triggerbase import trigger_base

####################################################
### FUNCTIONS IN THIS FILE HAVE NO TESTCASES YET ###
####################################################


def filter_chunk(data: np.ndarray, of: np.ndarray, record_length: int, pad: bool = False):
    """
    Filters 'data' with an optimum filter 'of' according to the overlap-add-algorithm.

    :param data: The data to filter.
    :type data: np.ndarray
    :param of: The filter to use.
    :type of: np.ndarray
    :param record_length: The record length as determined by the filter length (this determines, how many samples are discarded in the beginning and end of the data to avoid edge effects).
    :type record_length: int
    :param pad: Whether to zero-pad the filter kernel before usage.
    :type pad: bool

    :return: Filtered chunk
    :rtype: np.ndarray
    """
    # The scipy function 'oaconvolve' does NOT assume the response function to be in wrap-around order.
    # Therefore, we have to shift it such that the maximum position is again at 1/4 of the record window
    if pad:
        # I'm not entirely convinced that this is the correct way to do the padding. I think it should rather be rolled with -record_length//4, or rather identically to the phase multiplication below. However, if we roll and pad like this, we get results compatible with CAT.
        # Note that the filter normalization changes after padding (which in my mind makes no sense; hence, I think it is not fully correct). It does not change for roll(..., -record_length//4) but then the output doesn't look as nice. The filter normalization also changes in CAT. 
        # Therefore, this method should only be considered as a CAT cross check and comparison device.
        f = np.pad(
            np.roll(np.fft.irfft(of), record_length//4, axis=-1),
            (0, record_length),
        )
        # The padding here is chosen such that the slice of the oaconvolve output below
        # is identical to the non-padded scenario.
        data = np.pad(data[:-record_length//2], (record_length + record_length//2, 0))
    else:
        omega = 2 * np.pi * np.fft.rfftfreq(record_length)
        phase = np.exp(1j * record_length/4 * omega)
        f = np.fft.irfft(of*phase)

    return sp.signal.oaconvolve(
        f, 
        data, 
        mode="valid",
    )[record_length - record_length//4 + 1: -record_length//4]


def filter_chunk_2d(data: np.ndarray, of: np.ndarray, record_length: int):
    """
    Filters multi-channel 'data' with a 2D optimum filter 'of' according to the overlap-add-algorithm and sums them up to obtain a single, filtered trace, which is ready for triggering/maximum search.

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
    omega = 2 * np.pi * np.fft.rfftfreq(record_length)
    phase = np.exp(1j * record_length/4 * omega)
    return np.sum(
        np.atleast_2d(
            sp.signal.oaconvolve(
                np.fft.irfft(of*phase[None, ...]), data, mode="valid"
            )
        ),
        axis=0,
    )[record_length - record_length//4 + 1: -record_length//4]


def trigger_of(
    stream: ArrayLike,
    threshold: float,
    of: np.ndarray,
    pad: bool = False,
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
    :param pad: If True, the filter kernel is padded (in the time domain) before it is applied. You will/should rarely need this but it is implemented for comparisons to CAT. If you use it, make sure that your filter is correctly normalized for that usage. I.e., if `scale=np.max(vai.OptimumFiltering(of, method="linear_pad")(sev))` is not one, you should use a renormalized filter `of_pad=of/scale` instead.
    :type pad: bool
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
    filter_fnc = partial(filter_chunk, of=of, record_length=record_length, pad=pad)

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


def trigger_of2d(
    stream: ArrayLike,
    threshold: float,
    of: np.ndarray,
    n_triggers: int = None,
    chunk_size: int = 100,
    apply_first: Union[callable, List[callable]] = None,
    n_processes: int = None,
):
    """
    Trigger multiple channels of a stream at the same time using a 2D optimum filter and the optimum filter triggering algorithm described in https://edoc.ub.uni-muenchen.de/23762/. See :func:`cait.versatile.functions.trigger.triggerbase.trigger_base` for details on the implementation. The correlated filtering returns A SINGLE voltage trace (even for multiple input channels), hence, only a combined trigger threshold is required.

    Apart from the fact that multi-channel 'streams' and 'of2d' are required here, the usage is identical to :func:`cait.versatile.trigger_of`.

    :param stream: The stream channels to trigger (The 'channel' returns 2d-arrays. Usually, one would get this using ``stream[('Ch0', 'Ch1')]``).
    :type stream: ArrayLike
    :param threshold: The threshold (in Volts) above which events should be triggered.
    :type threshold: float
    :param of: The correlated (2d) optimum filter to be used for filtering (it is assumed that the filter's first entry is set to zero to correctly remove the offset).
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
    """
    # size of record window (as determined by the size of the filter)
    record_length = 2 * (of.shape[-1] - 1)

    # before samples exceeding threshold are searched, the chunks are filtered
    filter_fnc = partial(filter_chunk_2d, of=of, record_length=record_length)

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
