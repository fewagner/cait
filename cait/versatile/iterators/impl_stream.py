from typing import List, Union

import numpy as np

from .iteratorbase import IteratorBaseClass


def _validate_new_window(stream, name, new_alignment, new_record_length):
    ts = stream.timestamps
    dt = stream.dt_us
    # Timestamps of the first and last samples of 
    # the records currently in the iterator.
    record_start_ts = ts - dt * int(new_alignment * new_record_length)
    record_time = dt * new_record_length
    record_end_ts = record_start_ts + ( record_time - dt )
    
    # Check if any of the windows would extend outside the stream samples.
    flag_inside = (
        ( record_start_ts >= stream.ds_start_us )
        * ( record_end_ts <= stream._stream.time[-1] )
    )
    n_outside = np.sum(~flag_inside)

    if n_outside > 0:
        raise IndexError(f"If {name} was changed, {n_outside} of the events would have record windows extending outside of the valid sample range of the stream.")

class StreamIterator(IteratorBaseClass):
    """
    Iterator object that returns voltage traces for given trigger indices of a stream file. 

    :param stream: The stream object to read the voltage traces from.
    :type stream: StreamBaseClass
    :param keys: The keys (channel names) of the stream object to be iterated over. 
    :type keys: Union[str, List[str]]
    :param inds: The stream indices for which we want to read the voltage traces. This index is aligned according to 'alignment' (default: at 1/4th of the record window).
    :type inds: Union[int, List[int]]
    :param record_length: The number of samples to be returned for each index. Usually, those are powers of 2, e.g. 16384
    :type record_length: int
    :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index. Defaults to 1/4.
    :type alignment: float
    :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
    :type batch_size: int

    :return: Iterable object
    :rtype: StreamIterator
    """

    def __init__(self, 
                 stream, 
                 keys: Union[str, List[str]], 
                 inds: Union[int, List[int]], 
                 record_length: int, 
                 alignment: float = 1/4,
                 batch_size: int = None):
        
        if 0 > alignment or 1 < alignment:
            raise ValueError("'alignment' has to be in the interval [0,1]")
        
        self._keys = [keys] if isinstance(keys, str) else keys
        inds = [inds] if isinstance(inds, int) else [int(i) for i in inds]

        # Does batch handling and creates properties self._inds, self.uses_batches, and self.n_batches. 
        # Also sets up serializing.
        super().__init__(inds=inds, 
                         batch_size=batch_size, 
                         stream=stream, 
                         keys=keys, 
                         record_length=record_length, 
                         alignment=alignment)

        self._stream = stream
        self._record_length = record_length

        # Save values to reconstruct iterator:
        self._params = {'stream': stream, 
                        'keys': self._keys, 
                        'inds': inds, 
                        'record_length': record_length,
                        'alignment': alignment,
                        'batch_size': batch_size}

        self._alignment = alignment
        self._interval = (int(alignment*record_length), record_length - int(alignment*record_length))

    def __iter__(self):
        self._current_batch_ind = 0
        return self
    
    def __enter__(self):
        self._stream.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        self._stream.__exit__(typ, val, tb)

    def _next_raw(self):
        if self._current_batch_ind < self.n_batches:
            event_inds_in_batch = self._inds[self._current_batch_ind]
            self._current_batch_ind += 1

            if isinstance(event_inds_in_batch, int):
                s = slice(event_inds_in_batch - self._interval[0], 
                          event_inds_in_batch + self._interval[1])
                
                if len(self._keys) == 1:
                    out = self._stream[self._keys[0], s, 'as_voltage']
                else:
                    out = [self._stream[k, s, 'as_voltage'] for k in self._keys]
            
                return np.array(out)

            else:
                all_slices = np.r_.__getitem__(tuple(
                    [slice(i - self._interval[0], i + self._interval[1]) for i in event_inds_in_batch]
                    )
                ).reshape(len(event_inds_in_batch), self._record_length)

                if len(self._keys) == 1:
                    out = self._stream[self._keys[0], all_slices, 'as_voltage']
                else:
                    out = [self._stream[k, all_slices, 'as_voltage'] for k in self._keys]
                    out = np.transpose(np.array(out), axes=[1,0,2])
            
                return np.array(out)
            
        else:
            raise StopIteration
        
    def with_alignment(self, alignment: float):
        """
        Return an iterator for identical timestamps but with different alignment.

        :param alignment: A number in the interval [0,1] which determines the alignment of the record window (of length `record_length`) relative to the specified index. E.g. if `alignment=1/2`, the record window is centered around the index.
        :type alignment: float

        .. warning::
            Requires that all event traces are still within the stream boundaries after changing alignment.
        """
        if 0 > alignment or 1 < alignment:
            raise ValueError("'alignment' has to be in the interval [0,1]")
        
        if self.has_processing:
            raise NotImplementedError("Cannot change alignment for iterators with processing because processing might depend on alignment and cause obscure issues. Manually remove processing first using 'old_processing = it.pop_processing()', call 'with_alignment' on the iterator without processing, then add the 'old_processing' again if it does not depend on alignment, or add it again after adjusting its parameters to work with the new alignment.")
        
        _validate_new_window(self, "alignment", new_alignment=alignment, new_record_length=self.record_length)

        params, _ = self._slice_info
        new_params = params.copy()
        new_params["alignment"] = alignment

        return self.__class__(**new_params)

    def with_record_length(self, record_length: int):
        """
        Return an iterator for identical timestamps but with different record length.

        :param record_length: The number of samples to be returned for each event. Usually, those are powers of 2, e.g. 16384
        :type record_length: int

        .. warning::
            Requires that all event traces are still within the stream boundaries after changing record length.
        """
        if self.has_processing:
            raise NotImplementedError("Cannot change record_length for iterators with processing because processing might depend on window size and cause obscure issues. Manually remove processing first using 'old_processing = it.pop_processing()', call 'with_record_length' on the iterator without processing, then add the 'old_processing' again if it does not depend on window size, or add it again after adjusting its parameters to work with the new window size.")
        
        _validate_new_window(self, "record_length", new_alignment=self.alignment, new_record_length=record_length)
        
        params, _ = self._slice_info
        new_params = params.copy()
        new_params["record_length"] = record_length

        return self.__class__(**new_params)
    
    def with_extended_window(self):
        """Return an iterator for identical timestamps but with the window size increased to include one additional record length before and after the previous window."""
        return self.with_alignment(1 / 3 + self.alignment / 3).with_record_length(3 * self.record_length)
        
    @property
    def alignment(self):
        """
        The time axis alignment of the iterator. 
        
        For most event iterators, this is 1/4, i.e. the timestamp of an event corresponds to the sample at 1/4th of the record window. However, when constructing a StreamIterator, you may choose the alignment. Therefore, for StreamIterators, this value may be anything in the interval [0, 1].
        """
        return self._alignment
    
    # Overridden here because StreamIterator may define different alignment
    @property
    def t(self):
        """
        Return the time axis (record window) of the events in the iterator. 
        
        It is a millisecond array with 0 aligned according to the 'alignment' argument used when constructing the StreamIterator.
        """
        return (np.arange(self.record_length) - self.alignment*self.record_length)*self.dt_us/1000
    
    @property
    def record_length(self):
        return self._record_length
    
    @property
    def dt_us(self):
        return self._stream.dt_us
    
    @property
    def ds_start_us(self):
        return self._stream.start_us
    
    @property
    def timestamps(self):
        return self._stream.time[self._params["inds"]]
    
    @property
    def n_channels(self):
        return len(self._keys)
    
    @property
    def _slice_info(self):
        return (self._params, ('keys', 'inds'))