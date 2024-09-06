from typing import Union, List
import datetime

import numpy as np
import scipy as sp

from ..viewer import Viewer
from ...datasources.stream.streambase import StreamBaseClass
from ...datasources.stream.factory import Stream
from ...functions.trigger.trigger_of import filter_chunk

# Has no test case (yet)
class StreamViewer(Viewer):
    """
    Class to view stream data, i.e. for example the contents of a binary file as produced by vdaq2.

    :param args: Either an existing Stream instance or both 'hardware' (str) and file(s) (str or list of str).
    :type args: Union[StreamBaseClass, str, list]
    :param keys: The keys of the stream to display. If none are specified, all available keys are plotted. Defaults to None.
    :type keys: Union[str, List[str]]
    :param n_points: The number of data points that should be simultaneously displayed in the stream viewer. A large number can impact performance. Note that the number of points that are displayed are irrelevant of the downsampling factor (see below), i.e. the viewer will always display n_points points.
    :type n_points: int, optional
    :param downsample_factor: This many samples are skipped in the data when plotting it. A higher number increases performance but lowers the level of detail in the data.
    :type downsample_factor: int, optional
    :param mark_timestamps: A list of timestamps to be shown on top of the stream (e.g. to check trigger timestamps). Can also be a dictionary of lists, in which case they keys of the dictionary are used as legend entries.
    :type mark_timestamps: Union[List[int], int], optional
    :param of: If provided, a preview of the optimum filtered stream is shown. Only works for single-channel filters in which case also the 'keys' argument has to be set to exactly one channel (the one you want to filter).
    :type of: np.ndarray, optional
    :param kwargs: Keyword arguments for `Viewer`.
    :type kwargs: Any

    .. code-block:: python

        # Usage 1
        s = Stream(hardware="vdaq2", src="path/to/file.bin")
        StreamViewer(s)

        # Usage 2
        StreamViewer("vdaq2", "path/to/file.bin", key="ADC1")
    """
    def __init__(self, 
                 *args: Union[StreamBaseClass, str, list],
                 keys: Union[str, List[str]] = None,
                 n_points: int = 10000, 
                 downsample_factor: int = 100,
                 mark_timestamps: Union[List[int], dict] = None,
                 of: np.ndarray = None,
                 **kwargs):
        super().__init__(data=None, show_controls=True, **kwargs)

        # Adding buttons for navigating back and forth in the stream
        self._add_button("←", self._move_left, "Move backwards in time.", -1, "b")
        self._add_button("→", self._move_right, "Move forward in time.", -1, "n")

        if len(args) == 1 and isinstance(args[0], StreamBaseClass):
            self.stream = args[0]
        elif len(args) == 2:
            self.stream = Stream(hardware=args[0], src=args[1])
        else:
            raise ValueError(f"Invalid positional arguments '{args}'. Has to be either a StreamBaseClass instance or 'hardware' and 'files'.")

        if keys is not None:
            if type(keys) is str: 
                keys = [keys]
            if not all([k in self.stream.keys for k in keys]):
                raise KeyError("One or more keys are not present in the stream.")
            self._keys = keys
        else:
            self._keys = self.stream.keys

        # Adding lines
        for name in self._keys:
            self.add_line(x=None, y=None, name=name)

        # Adding labels
        # xlabel is dynamic
        self.set_ylabel("trace (V)")
        
        # Adding optimum filter
        if of is not None:
            if np.array(of).ndim > 1:
                raise ValueError(f"Only filtering of single channels is supported (i.e. 'of' has to be 1d).")
            if len(self._keys) > 1:
                raise ValueError(f"In case a filter is provided, you also have to choose a single channel (to be filtered) using the 'keys' argument.")
            
            self.add_line(x=None, y=None, name=f"{self._keys[0]} (filtered)")
                
        self._of = np.array(of) if of is not None else None

        # Adding timestamp markers
        if mark_timestamps is not None:
            if type(mark_timestamps) is not dict:
                mark_timestamps = dict(timestamps=np.array(mark_timestamps))
            self._marked_timestamps = mark_timestamps

            for name in mark_timestamps.keys():
                if len(mark_timestamps[name]) == 0:
                    raise Exception("Received empty list of timestamps")
                self.add_vmarker(marker_pos=None, y_int=[None,None], name=name)
            self._marks_timestamps = True
        else:
            self._marks_timestamps = False

        # Initializing plot
        self.n_points = n_points
        self.current_start = 0
        self.downsample_factor = downsample_factor

        self.update_frame()
        self.show()

    def update_frame(self):
        # Create slice for data access
        where = slice(self.current_start, self.current_start + self.n_points*self.downsample_factor, self.downsample_factor)
        
        # Time array is the same for all channels
        t = self.stream.time[where]

        # Find start and change x-label accordingly
        t_start = self.stream.time.timestamp_to_datetime(t[0])[None]
        t_str = t_start.astype(datetime.datetime)[0]
        self.set_xlabel(f"time (ms) after {t_str.strftime('%d-%b-%Y, %H:%M:%S')}, ({t[0]})")

        # Convert to milliseconds after first timestamp
        t_ms = (t-t[0])/1000

        val_min = []
        val_max = []
        
        for name in self._keys:
            y = self.stream[name, where, "as_voltage"]
            self.update_line(name=name, x=t_ms, y=y)

            if self._marks_timestamps:
                val_min.append(np.min(y))
                val_max.append(np.max(y))
                
        if self._of is not None:
            record_length = 2*(self._of.shape[-1] - 1)
            if self.current_start > record_length:
                where_filter = slice(self.current_start - record_length, 
                                 self.current_start + self.n_points*self.downsample_factor + record_length)
                chunk_to_filter = self.stream[self._keys[0], where_filter, "as_voltage"]
                
            else:
                where_filter = slice(self.current_start, 
                                 self.current_start + self.n_points*self.downsample_factor + record_length)
                chunk_to_filter = np.concatenate([np.zeros(record_length), self.stream[self._keys[0], where_filter, "as_voltage"]])
            
            
            
            filtered_stream = filter_chunk(chunk_to_filter, self._of, record_length)
            self.update_line(name=f"{self._keys[0]} (filtered)", x=t_ms, y=filtered_stream[::self.downsample_factor])
            
            if self._marks_timestamps:
                val_min.append(np.min(filtered_stream))
                val_max.append(np.max(filtered_stream))

        if self._marks_timestamps: 
            y_min, y_max = np.min(val_min), np.max(val_max)
            t_min, t_max = t[0], t[-1]

            for name in self._marked_timestamps.keys():
                mask = np.logical_and(np.array(self._marked_timestamps[name]) > t_min,
                                      np.array(self._marked_timestamps[name]) < t_max)
                ts = self._marked_timestamps[name][mask]
                
                if len(ts) > 0:
                    ts_ms = (ts-t[0])/1000
                else:
                    ts_ms = None
                
                self.update_vmarker(name=name, marker_pos=ts_ms, y_int=(y_min, y_max))

        self.update()

    def _move_right(self, b=None):
        # ATTENTION: should be restricted to file size at some point (and the end point should be provided by stream)
        self.current_start += int(self.n_points*self.downsample_factor/2)
        self.update_frame()

    def _move_left(self, b=None):
        self.current_start = max(0, self.current_start - int(self.n_points*self.downsample_factor/2))
        self.update_frame()