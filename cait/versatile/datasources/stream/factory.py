from typing import Union, List

from .streambase import StreamBaseClass
from .impl_cresst import Stream_CRESST
from .impl_vdaq2 import Stream_VDAQ2

class Stream(StreamBaseClass):
    """
    Factory class for providing a common access point to stream data.
    Currently, only vdaq2 and cresst stream files are supported but an extension can be straight forwardly implemented by sub-classing :class:`StreamBaseClass` and adding it for selection in the constructor of :class:`Stream`.

    The data is accessed by means of slicing (see below). The `time` property is an object of :class:`StreamTime` and offers a convenient time interface as well (see below).

    :param hardware: The hardware which was used to record the stream file. Valid options are ['cresst', 'vdaq2']
    :type hardware: str
    :param src: The source for the stream. Depending on how the data is taken, this can either be the path to one file or a list of paths to multiple files. This input is handled by the specific implementation of the Stream Object. See below for examples.
    :type src: Union[str, List[str]]

    **Usage for different hardware:**
    
    CRESST:
    Files are .csmpl files which contain one channel each. Additionally, we need a .par file to read the start timestamp of the stream data from.
    ::
        s = Stream(hardware='cresst', src=['par_file.par', 'stream_Ch0.csmpl', 'stream_Ch1.csmpl'])

    VDAQ2:
    Files are .bin files which contain all information necessary to construct the Stream object. It can be input as a single argument.
    ::
        s = Stream(hardware='vdaq2', src='file.bin')

    **Usage slicing:**

    Valid options for slicing streams are the following:
    ::
        # Get data for one channel
        s['ADC1']

        # Get data for one channel and slice it (two equivalent ways)
        s['ADC1', 10:20]
        s['ADC1'][10:20]

        # Get data for one channel, slice it, and return the voltage 
        # values instead of the ADC values
        s['ADC1', 10:20, 'as_voltage']
    """
    def __init__(self, hardware: str, src: Union[str, List[str]]):
        if hardware.lower() == "cresst":
            self._stream = Stream_CRESST(src)
        elif hardware.lower() == "vdaq2":
            self._stream = Stream_VDAQ2(src)
        #elif hardware.lower() == "vdaq3":
        #    self._stream = Stream_VDAQ3(src)
        else:
            raise NotImplementedError('Only cresst and vdaq2 files are supported at the moment.')

    def __repr__(self):
        return repr(self._stream)
    
    def __len__(self):
        return len(self._stream)
        
    def get_channel(self, key: str):
        return self._stream.get_channel(key)
        
    def get_voltage_trace(self, key: str, where: slice):
        return self._stream.get_voltage_trace(key, where)

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
    def tpas(self):
        return self._stream.tpas

    @property
    def tp_timestamps(self):
        return self._stream.tp_timestamps