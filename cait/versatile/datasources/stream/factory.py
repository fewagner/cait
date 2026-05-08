from typing import List, Union

from .impl_csmpl import Stream_CSMPL
from .impl_vdaq2 import Stream_VDAQ2
from .impl_vdaq3 import Stream_VDAQ3
from .streambase import StreamBaseClass


class Stream(StreamBaseClass):
    """
    Factory class for providing a common access point to stream data.
    Currently, only vdaq3, vdaq2 and csmpl stream files are supported but an extension can be straight forwardly implemented by sub-classing :class:`cait.versatile.datasources.stream.streambase.StreamBaseClass` and adding it for selection in the constructor of :class:`Stream`.

    The data is accessed by means of slicing (see below). The `time` property is an object of :class:`StreamTime` and offers a convenient time interface as well (see below).

    **Important Note:** If you plan to access the stream data repeatedly, you can ensure that the stream file stays open (increases speed) by using it as a context manager:

    .. code-block:: python

        import cait.versatile as vai

        stream = vai.Stream(hardware='vdaq2', src='file.bin')
        with stream:
            trigger_inds, amplitudes = vai.trigger_zscore(stream["ADC1"], 2**14)

    :param hardware: The hardware which was used to record the stream file. Valid options are ['csmpl', 'vdaq2', 'vdaq3']
    :type hardware: str
    :param src: The source for the stream. Depending on how the data is taken, this can either be the path to one file or a list of paths to multiple files. This input is handled by the specific implementation of the Stream Object. See below for examples.
    :type src: Union[str, List[str]]
    :param args, kwargs: Additional arguments for the chosen hardware (see respective documentation).
    :type args, kwargs: Any

    **Usage for different hardware:**

    CSMPL:
    Files are ``.csmpl`` files which contain one channel each. Additionally, we need a ``.par`` file to read the start timestamp of the stream data from.

    .. code-block:: python

        s = Stream(hardware='csmpl', src=['par_file.par', 'stream_Ch0.csmpl', 'stream_Ch1.csmpl'])

    See also: :class:`cait.versatile.datasources.stream.impl_csmpl.Stream_CSMPL`

    VDAQ2:
    Files are ``.bin`` files which contain all information necessary to construct the Stream object. It can be input as a single argument. Testpulse channels in this file format need to be (automatically) triggered to obtain testpulse amplitudes and timestamps.

    .. code-block:: python

        s = Stream(hardware='vdaq2', src='file.bin')

    See also: :class:`cait.versatile.datasources.stream.impl_vdaq2.Stream_VDAQ2`

    VDAQ3:
    Files are ``.bin`` files which contain one channel each. There are two versions of the file format: One for which the testpulse timestamps are already saved inside the ``.bin`` file (preferred format), and one for which you have to load the testpulse channel as an additional stream channel and (automatically) trigger them to get the timestamps/tpas (like for the VDAQ2 format).

    .. code-block:: python

        s = Stream(hardware='vdaq3', src=['file_ch0.bin', 'file_ch1.bin'])

    See also: :class:`cait.versatile.datasources.stream.impl_vdaq3.Stream_VDAQ3`

    **Usage slicing:**

    Valid options for slicing streams are the following:

    .. code-block:: python

        # Get voltage data for one channel (this does NOT load it
        # into memory but you can use the resulting object, more or
        # less, like a numpy-array).
        ch1 = s['ADC1']
        ch2 = s['ADC2']

        # This also works for multiple channels. Note, however, that
        # you still slice it as if it was 1d, i.e. if you slice the
        # first 10 elements of the object, you will get the first 10
        # for BOTH channels.
        chs = s[['ADC1', 'ADC2']]
        chs[:10] # equivalent to np.array([ch1[:10], ch2[:10]])

        # Get ADC data for one channel and slice it (two equivalent ways)
        s['ADC1', 10:20]
        s['ADC1'][10:20]

        # Get voltage data for one channel, slice it, and return the
        # voltage values instead of the ADC values. The cleaner way
        # to do this would be to use the first syntax above.
        s['ADC1', 10:20, 'as_voltage']
    """

    def __init__(self, hardware: str, src: Union[str, List[str]], *args, **kwargs):
        super().__init__(hardware, src, *args, **kwargs)

        if hardware.lower() == "csmpl":
            self._stream = Stream_CSMPL(src, *args, **kwargs)
        elif hardware.lower() == "vdaq2":
            self._stream = Stream_VDAQ2(src, *args, **kwargs)
        elif hardware.lower() == "vdaq3":
           self._stream = Stream_VDAQ3(src, *args, **kwargs)
        else:
            raise NotImplementedError(
                "Only csmpl, vdaq2, and vdaq3 files are supported at the moment."
            )

    def __repr__(self):
        return repr(self._stream)

    def __len__(self):
        return len(self._stream)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, typ, val, tb):
        self._stream.__exit__(typ, val, tb)

    def get_trace(self, key: str, where: slice, voltage: bool = True):
        return self._stream.get_trace(key, where, voltage=voltage)

    # redirect all attribute calls to underlying stream object
    # (if not explicitly defined by this class)
    def __getattr__(self, name):
        if hasattr(self._stream, name):
            return self._stream.__getattribute__(name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'.")
        
    @property
    def keys(self):
        return self._stream.keys

    @property
    def start_us(self):
        return self._stream.start_us

    @property
    def dt_us(self):
        return self._stream.dt_us

    @property
    def tp_keys(self):
        return self._stream.tp_keys

    @property
    def tpas(self):
        return self._stream.tpas

    @property
    def tp_timestamps(self):
        return self._stream.tp_timestamps
    
    @property
    def calp_keys(self):
        return self._stream.calp_keys

    @property
    def calpas(self):
        return self._stream.calpas

    @property
    def calp_timestamps(self):
        return self._stream.calp_timestamps
