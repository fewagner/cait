import os

from typing import Union

import numpy as np
import cait as ai

from .par_file import PARFile
from ..datasourcebase import DataSourceBaseClass
from ...iterators.impl_rdt import RDTIterator

class RDTFile:
    """
    Class for interfacing hardware triggered files (file extension `.rdt`). This class automatically infers the available channels and the available correlated channels. Those can be retrieved by indexing the RDTFile object with channel indices/names or tuples thereof, the result of the indexing is a :class:`RDTChannel` object which provides testpulse amplitudes, timestamps, and event iterators for (the) selected channel(s) (see documentation for :class:`RDTChannel`).

    :param path: The full path (including the file extension `.rdt`) to the file of interest.
    :type path: str
    :param path_par: The full path (including the file extension `.par`) to the file which contains the necessary parameters to read the `.rdt` file. If None is given, it is assumed that a `.par` file with identical name/path as `path` is available. Defaults to None.
    :type path_par: str, optional

    :return: Object interfacing an `.rdt` file.
    :rtype: RDTFile

    **Example:**
    ::
        import cait.versatile as vai

        f = vai.RDTFile('path/to/file.rdt')

        # Check available channels
        print(f.keys)
        # Choose channel(s) to iterate over, get testpulse amplitudes, ...,  by slicing RDTFile
        channels = f[(0,1)] # if interested in only one channel: channel0 = f[0]
        it = channels.get_event_iterator()

        # You can now further slice this iterator (like any other iterator in cait.versatile):
        it_testpulses = it[:, channels.tpas > 0]
        it_events = it[:, channels.tpas == 0]
        it_noise = it[:, channels.tpas == -1]

        # Have a look (after removing the baseline):
        vai.Preview(it_testpulses.with_processing(vai.RemoveBaseline()))
    """
    def __init__(self, path: str, path_par: str = None):
        if not path.endswith(".rdt"):
            raise ValueError("Unrecognized file extension for 'path'. Please input an *.rdt file.")
        
        if path_par is None:
            path_par = os.path.splitext(path)[0] + '.par'

        if not path_par.endswith(".par"):
            raise ValueError("Unrecognized file extension 'path_par'. Please input a *.par file.")

        self._par = PARFile(path_par)

        self._dtype = np.dtype([
                       ('detector_nmbr', 'i4'),
                       ('coincide_pulses', 'i4'),
                       ('trig_count', 'i4'),
                       ('trig_delay', 'i4'),
                       ('abs_time_s', 'i4'),
                       ('abs_time_mus', 'i4'),
                       ('delay_ch_tp', 'i4', (int(self._par.ints_in_header == 7),)),
                       ('time_low', 'i4'),
                       ('time_high', 'i4'),
                       ('qcd_events', 'i4'),
                       ('hours', 'f4'),
                       ('dead_time', 'f4'),
                       ('test_pulse_amplitude', 'f4'),
                       ('dac_output', 'f4'),
                       ('dvm_channels', 'f4', self._par.dvm_channels),
                       ('samples', 'i2', self._par.record_length),
                       ])
        
        self._raw_file = np.memmap(path, dtype=self._dtype, mode='r')

        # Copy the relevant data to memory so that we don't have to go through
        # the memory mapped file all the time (this should only need a few MB of RAM)
        meta_copy = np.array(self._raw_file[["detector_nmbr", "trig_count"]])

        self._available_channels = np.unique(meta_copy["detector_nmbr"]).tolist()

        # I'M STILL NOT SURE IF I WANT A DEFAULT CHANNELS BEHAVIOR OR NOT 
        # If no channels are requested by the user (through __getitem__), the default behavior is such
        # that all quantities (iterators, timestamps, tpas) are returned for all channels.
        # Notice that this might fail (when not all are correlated). 
        # In the case that, e.g., only two channels are present and correlated, this provides a shortcut
        # when accessing the important stuff
        # if len(self._available_channels) > 1:
        #     self._default_channels = tuple(self._available_channels)
        # else:
        #     self._default_channels = int(self._available_channels[0])

        self._inds = dict()
        # DETERMINE INDICES FOR SINGLE CHANNELS:
        for c in self._available_channels:
            self._inds[c] = np.nonzero(meta_copy["detector_nmbr"] == c)[0]

        # DETERMINE INDICES FOR CORRELATED CHANNELS (have same trig_count):
        vals, idx_start, count = np.unique(meta_copy["trig_count"], return_counts=True, return_index=True)
        
        # We are only interested in trigger counts that appear more than once
        flag_corr = count > 1
        if np.sum(flag_corr) > 0:

            # Split trigger count array according to the first occurences
            # (note: the first entry is discarded)
            s = np.split(meta_copy["trig_count"], idx_start)[1:]
        
            # Now restrict to those cases which include more than one occurrence 
            # (note that this has to be done AFTER the splitting above)
            vals = vals[flag_corr]
            idx_start = idx_start[flag_corr]
            count = count[flag_corr]

            # Furthermore, we are only interested in identical trigger counts that
            # are adjacent in the file (ensure that only correctly written records
            # are included)
            # (The next line just checks if the pieces in s all include the same value, i.e. are consecutive)
            flag_adj = [len(np.unique(x))==1 for x in s if len(x)>1]
            
            # Restrict to adjacent
            vals = vals[flag_adj]
            idx_start = idx_start[flag_adj]
            count = count[flag_adj]

            # Create and populate a dictionary whose keys are the unique 
            # channel combinations and whose values are the corresponding 
            # data indices
            unique_tuples = dict()
            for n, (i, l) in enumerate(zip(idx_start, count)):
                tup = tuple(meta_copy["detector_nmbr"][i:(i+l)])
                if not (tup in unique_tuples.keys()): 
                    unique_tuples[tup] = []
                unique_tuples[tup].append(n)
                
            # Add tuples and corresponding indices to self._inds dictionary
            for k, v in unique_tuples.items():
                tup_inds = np.array(v)
                self._inds[k] = idx_start[tup_inds]
        
    def __repr__(self):
        k = self.keys.keys() if isinstance(self.keys, dict) else self.keys
        return f'{self.__class__.__name__}(keys={k}, record_length={self.record_length}, dt_us={self.dt_us}, measuring_time_h={self.measuring_time_h:.2f})'

    def __getitem__(self, channels: Union[int, str, tuple]):
        """
        Choose a single channel (integer) or correlated channels (tuple of integers) from this RDTFile for further investigations. 
        The available keys are found via `RDTFile.keys`. If the channels have names (successfully extracted from the `*.par` file), you can also use those for indexing.

        :param channels: Channel index (potentially name) or tuple thereof.
        :type channels: Union[int, str, tuple]

        :return: RDTChannel instance corresponding to the choice of channels
        :rtype: RDTChannel
        """
        if isinstance(channels, str):
            if not self._par.has_channel_names:
                raise KeyError("Indexing with strings is not supported because no channel name information is available for this instance.")
            # If channel names are available, we use the dictionary to find the corresponding keys
            corresponding_key = [k for k,v in self.keys.items() if v == channels]
            if not corresponding_key:
                raise KeyError(f"Name '{channels}' is not a valid channel name. Did you mean {[v for k,v in self.keys.items()]}")
            # If channel names are available and corresponding key could be retrieved, we switch to channel numbers
            channels = corresponding_key[0]
        elif isinstance(channels, tuple):
            l = list(channels)
            for i, c in enumerate(l):
                if isinstance(c, str):
                    if not self._par.has_channel_names:
                        raise KeyError("Indexing with strings is not supported because no channel name information is available for this instance.")
                    corresponding_key = [k for k,v in self.keys.items() if v == c]
                    if not corresponding_key:
                        raise KeyError(f"Name '{c}' is not a valid channel name. Did you mean {[v for k,v in self.keys.items()]}")
                    l[i] = corresponding_key[0]

            channels = tuple(l)
        elif not isinstance(channels, int):
            raise TypeError(f"Unsupported input type {type(channels)} for argument 'channels'.")
        
        if not channels in self._inds.keys():
            raise KeyError(f"'{channels}' is not available. Did you mean {', '.join([str(x) for x in self._inds.keys()])}?")
        
        return RDTChannel(self, key=channels)
    
    @property
    def _file(self):
        """The `numpy.memmap` object to the underlying `*.rdt` file."""
        return self._raw_file
    
    @property
    def record_length(self):
        """The record length (number of samples per event) of the events in the corresponding `*.rdt` file."""
        return self._par.record_length
    
    @property
    def dt_us(self):
        """The time base in microseconds (time between two samples) of the events in the corresponding `*.rdt` file."""
        return self._par.time_base_us
    
    @property
    def sample_frequency(self):
        """The sample frequency in Hz of the events in the corresponding `*.rdt` file."""
        return int(np.round(1e6/self._par.time_base_us))
    
    @property
    def measuring_time_h(self):
        """The total measuring time in hours of the corresponding `*.rdt` file."""
        return self._par.measuring_time_h
    
    @property
    def keys(self):
        """The channel keys that can be used to index this RDTFile instance. If available, the channel names (corresponding to the indices) are shown as well."""
        if self._par.has_channel_names:
            d = dict()
            for k in self._inds.keys():
                if isinstance(k, tuple):
                    d[k] = tuple([self._par.channel_names[i] for i in k])
                else: # then int
                    d[k] = self._par.channel_names[k]
                    
            return d
        else:
            return list(self._inds.keys())
        
    def get_voltage_trace(self, inds: Union[int, list]):
        """
        Return the voltage traces of events in this RDTFile for given indices.
        
        :param inds: The indices for which to return the voltage traces.
        :type inds: Union[int, list]

        :return: Array of as many voltage traces as given `inds`.
        :rtype: numpy.array
        """
        return ai.data.convert_to_V(self._file["samples"][inds], bits=16, min=-10, max=10)

    # I'M STILL NOT SURE IF I WANT A DEFAULT CHANNELS BEHAVIOR OR NOT     
    # @property
    # def default_channels(self):
    #     """
    #     This RDTFile's default channel(s). 
    #     Those are used if `tpas`, `unique_tpas`, `timestamps` and `get_event_iterator` are called on this RDTFile instance.
    #     """
    #     return self._default_channels
    
    # @property
    # def timestamps(self):
    #     """The microsecond timestamps of the events in this RDTFile's default channel(s)."""
    #     # Create an RDTChannel instance for the default channels and return its timestamps
    #     return self[self._default_channels].timestamps
    
    # @property
    # def tpas(self):
    #     """The testpulse amplitudes of the events in this RDTFile's default channel(s)."""
    #     # Create an RDTChannel instance for the default channels and return its tpas
    #     return self[self._default_channels].tpas
    
    # @property
    # def unique_tpas(self):
    #     """The unique testpulse amplitudes of the events in this RDTFile's default channel(s)."""
    #     # Create an RDTChannel instance for the default channels and return its unique_tpas
    #     return self[self._default_channels].unique_tpas
    
    # def get_event_iterator(self):
    #     """
    #     Get an iterator over the events of this RDTFile's default channel(s). 
    #     Note that this is merely a shortcut to first choosing the channel of interest and then constructing the iterator, and that the recommended way is to NOT use this shortcut unless you are certain that you want to use the default channel.

    #     :return: Iterable object
    #     :rtype: RDTIterator

    #     >>> import cait.versatile as vai
    #     >>> f = vai.RDTFile('path/to/file.rdt')
    #     >>> # Check what default channel is:
    #     >>> print(f.default_channel)
    #     >>> # If you're happy with the default channel, use f.get_event_iterator()
    #     >>> # If you'd rather choose another channel, you can do so by slicing, 
    #     >>> # e.g., f[0] for the channel number 0
    #     >>> it = f.get_event_iterator()
    #     >>> # You can now further slice this iterator (like any other iterator in cait.versatile):
    #     >>> it_testpulses = it[:, f.tpas > 0]
    #     >>> it_events = it[:, f.tpas == 0]
    #     >>> it_noise = it[:, f.tpas == -1]
    #     >>> # Remove baselines:
    #     >>> it_testpulses.add_processing(vai.RemoveBaseline())
    #     >>> # Have a look:
    #     >>> vai.Preview(it_testpulses)
    #     """
    #     # Create an RDTChannel instance for the default channels and return its event_iterator
    #     return self[self._default_channels].get_event_iterator()
        
class RDTChannel(DataSourceBaseClass):
    """
    Object representing a coherent part of an RDTFile (i.e. either a single channel or correlated channels). Usually this is not created as a standalone but the result of slicing an RDTFile.

    :param rdt_file: An RDTFile instance.
    :type rdt_file: RDTFile
    :param key: The key which selects the single channel or correlated channels. Either of `rdt_file.keys`.
    :type key: Union[int, tuple]

    :return: Specified channels of an RDTFile
    :rtype: RDTChannel
    """
    def __init__(self, rdt_file: RDTFile, key: Union[int, tuple]):

        inds = rdt_file._inds[key]
        self._n_channels = len(key) if isinstance(key, tuple) else 1

        # Build index array for all channels, first dimension is channel index (also existent if only one
        # channel is present). Slicing this array then corresponds to choosing channel(s)/event(s) in 
        # original file
        self._inds = np.array([np.array(inds)+k for k in range(self._n_channels)])

        self._key = key
        self._rdt_file = rdt_file

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self._key}, n_events={len(self)}, n_channels={self._n_channels}, unique_tpas={self.unique_tpas})"
    
    def __len__(self):
        return self._inds.shape[-1]
    
    def __getitem__(self, val):
        """Return voltage traces of events. Any numpy indexing can be used as if it was an array of shape (n_channels, n_events)."""
        if isinstance(val, tuple):
            if len(val) != 2: 
                raise ValueError("Indexing only supports up to two arguments.")
            requested_events = self._inds[val[0]][...,val[1]].T
        else:
            requested_events = self._inds[..., val].T

        return self._rdt_file.get_voltage_trace(requested_events)
    
    @property
    def key(self):
        """The RDTFile key that this RDTChannel corresponds to."""
        return self._key
    
    @property
    def n_channels(self):
        """The number of channels this RDTChannel corresponds to."""
        return self._n_channels
    
    @property
    def start_us(self):
        s = self._rdt_file._par.start_s
        mus = self._rdt_file._par.start_us
        return s*int(1e6) + mus
    
    @property
    def timestamps(self):
        """The microsecond timestamps of the events in this RDTChannel."""
        secs = np.array(self._rdt_file._file["abs_time_s"][self._inds[0]], dtype=np.int64)
        msecs = np.array(self._rdt_file._file["abs_time_mus"][self._inds[0]], dtype=np.int64)

        return secs*int(1e6) + msecs
    
    @property
    def tpas(self):
        """The testpulse amplitudes of the events in this RDTChannel."""
        return self._rdt_file._file["test_pulse_amplitude"][self._inds[0]]
    
    @property
    def unique_tpas(self):
        """The unique testpulse amplitudes of the events in this RDTChannel."""
        return sorted(list(set(self.tpas)))
    
    def get_event_iterator(self, batch_size: int = None):
        """
        Get an iterator over the events present in this RDTChannel instance. 

        :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
        :type batch_size: int

        :return: Iterable object
        :rtype: RDTIterator

        **Example:**
        ::
            import cait.versatile as vai

            f = vai.RDTFile('path/to/file.rdt')

            # Choose channel(s) to iterate over by slicing RDTFile
            channels = f[(0,1)]
            it = channels.get_event_iterator()

            # You can now further slice this iterator (like any other iterator in cait.versatile):
            it_testpulses = it[:, channels.tpas > 0]
            it_events = it[:, channels.tpas == 0]
            it_noise = it[:, channels.tpas == -1]
            # Remove baselines:
            it_testpulses.add_processing(vai.RemoveBaseline())
            
            # Have a look:
            vai.Preview(it_testpulses)
        """
        return RDTIterator(self, batch_size=batch_size)