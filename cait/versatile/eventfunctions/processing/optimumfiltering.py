import numpy as np
import scipy as sp

from ..functionbase import FncBaseClass


def _sanitize_filter_shape(event, f):
    if event.ndim == 3:
        required_filter_shape = (
            *event.shape[0:2], 
            f.shape[-1],
        )
    elif event.ndim == 2:
        required_filter_shape = (
            event.shape[0], 
            f.shape[-1],
        )
    else:
        required_filter_shape = (
            1, 
            f.shape[-1],
        )

    return np.broadcast_to(f, required_filter_shape)


class OptimumFiltering(FncBaseClass):
    """
    Apply an optimum filter to a voltage trace.
    Works for multiple channels simultaneously if optimum filter is also given for multiple channels.

    :param of: The optimum filter to use.
    :type of: np.ndarray
    :param method: The method by which the filter is convolved with an event. Can be either 'circular' in which case the event is Fourier transformed, multiplied by the filter kernel, and back transformed, 'linear' in which case the event is convolved without warp-around (samples which would require wrap-around are removed), and 'linear_pad' which is the same as 'linear' but the trace is first zero-padded (retaining the same number of samples in the output trace). 'linear' and 'linear_pad' also support events that are larger than the filter kernel permits in the 'circular' case. Defaults to 'circular'.
    :type method: str

    :return: Filtered event.
    :rtype: np.ndarray

    .. warning::
        For filters that were built from a SEV whose maximum is not aligned at 1/4th of the record window, methods 'circular' and 'linear' might yield different results. In particular, the peak positions of the filtered traces might be offset. When using 'linear', it is recommended to have the SEV's maximum aligned at 1/4th of the record window.

    .. warning::
        When choosing to zero-pad your events (method 'linear_pad'), you should first remove any constant baseline from your event traces (``it.with_processing(vai.RemoveBaseline())``).

    **Example:**

    .. code-block:: python

        import cait.versatile as vai

        # Construct mock data (which provides event iterator and optimum filter)
        md = vai.MockData()
        it = md.get_event_iterator()[0].with_processing(vai.RemoveBaseline())
        of = md.of[0]

        # View effect of filtering on events
        vai.Preview(it, vai.OptimumFiltering(of))

    .. image:: media/OptimumFiltering_preview.png
    """
    _supported_methods = ["circular", "linear", "linear_pad"]
    def __init__(self, of: np.ndarray, method: str = "circular"):
        if method.lower() not in self._supported_methods:
            raise NotImplementedError(f"Method '{method}' is not supported. Choose either of {self._supported_methods}")
        
        self._of = np.atleast_2d(of)

        if method.lower() == "circular":
            self._call_f = self._impl_multiply
        elif method.lower().startswith("linear"):
            self._pad = method.lower().endswith("_pad")
            self._rl = 2 * (self._of.shape[-1] - 1)
            omega = 2 * np.pi * np.fft.rfftfreq(self._rl)
            self._of_time_domain = np.fft.irfft(self._of * np.exp(1j * self._rl/4 * omega))
            self._call_f = self._impl_slide

        self._method = method.lower()

    def __call__(self, event):
        return self._call_f(event)

    def _impl_multiply(self, event):
        in_shape = np.shape(event)
        event = np.atleast_2d(event)

        if (
            (self._of.shape[0] > 1 and (event.shape[-2] != self._of.shape[0]))
            or (self._of.shape[-1] != event.shape[-1]//2 + 1)
        ):
            raise ValueError(
                f"Shape mismatch of OF ({self._of.shape}) and event ({event.shape}). For filters of shape (K, N//2+1), where K is the number of channels, the event must have dimension (L, K, N)."
            )

        return np.reshape(
            np.fft.irfft(np.fft.rfft(event) * _sanitize_filter_shape(event, self._of)), 
            in_shape
        )

    def _impl_slide(self, event):
        in_shape = np.shape(event)
        event = np.atleast_2d(event)

        if (
            (self._of.shape[0] == 1 and event.ndim == 3)
            or (self._of.shape[0] > 1 and (event.shape[-2] != self._of.shape[0]))
        ):
            raise ValueError(
                f"Shape mismatch of OF ({self._of.shape}) and event ({event.shape}). For filters of shape (K, ...), where K is the number of channels, the event must have dimension (L, K, ...)."
            )

        if self._pad:
            k = event.ndim - 1
            event = np.pad(event, (*([(0, 0)]*k), (self._rl, self._rl)))
            reshape = lambda x: np.reshape(x, in_shape)
        else:    
            reshape = lambda x: np.reshape(x, (*in_shape[:-1], in_shape[-1] - 2*self._rl))

        # Even though it is technically fine to have a linear convolution with as few as 
        # record_length samples (gives one output sample), our alignment procedure and
        # the restriction to have at least record_length output samples forces us to use
        # more samples.
        if event.shape[-1] < 3 * self._rl:
            raise ValueError(f"For linear convolution and filters of length {self._of.shape[-1]}, the minimum event length is {3 * self._rl}. Got {event.shape[-1]} (which includes potential zero-padding when choosing method='linear_pad').")
        
        return reshape(
            sp.signal.oaconvolve(
                _sanitize_filter_shape(event, self._of_time_domain), 
                event, 
                mode="valid",
                axes=-1,
            )[..., self._rl - self._rl//4 + 1: -self._rl//4]
        )

    @property
    def batch_support(self):
        return "full"

    def preview(self, event) -> dict:
        filtered_event = self(event)
        filtered_x = np.arange(event.shape[-1])

        if self._method == "linear":
            filtered_x = filtered_x[self._rl: -self._rl]

        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f"channel {i}"] = [None, event[i]]
                d[f"filtered channel {i}"] = [filtered_x, filtered_event[i]]
        else:
            d = {"event": [None, event], "filtered event": [filtered_x, filtered_event]}
        return dict(line=d)


class OptimumFiltering2D(OptimumFiltering):
    """
    Apply a 2D optimum filter to multi-channel voltage traces.
    Regardless of the number of input channels, this function always returns a single, combined channel result.

    Equivalent to regular :class:`OptimumFiltering` all the channels and summing the result. In general, this only makes sense if you built the multi-channel OF from a Noise Covariance Matrix (:class:`cait.versatile.NCM`), i.e. when the OF incorporates information from all the channels.

    :param of: The optimum filter to use.
    :type of: np.ndarray
    :param kwargs: Additional keyword arguments for :class:`cait.versatile.OptimumFiltering`.
    :type kwargs: Any

    :return: Filtered event.
    :rtype: np.ndarray
    """

    def __init__(self, of: np.ndarray, **kwargs):
        if np.atleast_2d(of).shape[0] < 2:
            raise ValueError(f"Applying a 2D optimum filter requires filters with at least two channels. Got {np.atleast_2d(of).shape[0]}.")
        
        super().__init__(of, **kwargs)

    def __call__(self, event):
        return np.sum(np.atleast_2d(super().__call__(event)), axis=-2)

    def preview(self, event) -> dict:
        filtered_event = self(event)
        filtered_x = np.arange(event.shape[-1])

        if self._method == "linear":
            filtered_x = filtered_x[self._rl: -self._rl]

        if np.ndim(event) > 1:
            d = dict()
            for i in range(np.ndim(event)):
                d[f"channel {i}"] = [None, event[i]]
        else:
            d = {"event": [None, event]}

        d["filtered event"] = [filtered_x, filtered_event]

        return dict(line=d)