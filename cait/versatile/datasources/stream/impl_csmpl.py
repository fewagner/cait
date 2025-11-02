import json
import os
from typing import List, Union

import numpy as np

import cait as ai

from ....readers import BinaryFile
from ..hardwaretriggered.par_file import PARFile
from .streambase import StreamBaseClass
from ...functions import trigger_zscore, apply
from ...eventfunctions import MainParameters


def get_offset(path_dig_stamps):
    """
    Get the offset between start of the continuous DAQ and start of the CCS time recording.

    :param path_dig_stamps: The full path to the `*.dig` file.
    :type path_dig_stamps: str
    :return: The offset that needs to be subtracted from all CCS time stamps, to get the time stamps w.r.t. the start of the CSMPL file.
    :rtype: int
    """

    dig = np.dtype([
        ('stamp', np.uint64),
        ('bank', np.uint32),
        ('bank2', np.uint32),
    ])

    #diq_stamps = np.fromfile(path_dig_stamps, dtype=dig)
    diq_stamps = BinaryFile(path=path_dig_stamps, dtype=dig)
    dig_samples = diq_stamps['stamp']
    offset_clock = (dig_samples[1] - 2 * dig_samples[0])

    return offset_clock

def get_test_stamps(path,
                    channels=None,
                    control_pulses=None,
                    clock=10000000,
                    min_cpa=10.1):
    """
    Load the test pulse time stamps from a ``*.test_stamps`` file.

    :param path: The path to the ``*.test_stamps`` file.
    :type path: string
    :param channels: The test pulse channels we want to read out.
    :type channels: list
    :param control_pulses: If set to True, only control pulses are returned. If False, only test pulses are returned. If None, all are returned.
    :type control_pulses: bool or None
    :param clock: The Frequency of the time clock, in Hz. Standard for CRESST is 10MHz.
    :type clock: int
    :return: (the test pulse hours time stamps, the test pulse amplitudes, the channels of the test pulses)
    :rtype: 3-tuple of 1D arrays
    """

    teststamp = np.dtype([
        ('stamp', np.uint64),
        ('tpa', np.float32),
        ('tpch', np.uint32),
    ])

    #stamps = np.fromfile(path, dtype=teststamp)
    stamps = BinaryFile(path=path, dtype=teststamp)

    hours = stamps['stamp'] / clock / 3600
    tpas = stamps['tpa']
    testpulse_channels = stamps['tpch']

    # take only the channels we want
    if channels is not None:
        # Deprecated since numpy 2.0
        # cond = np.in1d(testpulse_channels, channels)
        cond = np.isin(testpulse_channels, channels)
        hours = hours[cond]
        tpas = tpas[cond]
        testpulse_channels = testpulse_channels[cond]

    # take only control or no control pulses
    if control_pulses is not None:
        if control_pulses:
            cond = tpas > min_cpa
        else:
            cond = tpas < min_cpa
        hours = hours[cond]
        tpas = tpas[cond]
        testpulse_channels = testpulse_channels[cond]

    return hours, tpas, testpulse_channels


def get_tpa_from_stream(ph, tpas, threshold):
    """
    Get the most likely TP amplitude that was sent, from the pulse height of
    an event in the testpulse stream.  If the "confidence" (the ratio of the
    difference between the pulse height and most likely TPA to the rest of
    the TPAs) is greater than the threshold for any other TPA, then the event
    is TPA inconclusive and the value is set to -1.

    Note that the pulse height should not have been calculated from a smoothed
    stream.

    :param ph: Pulse height from the triggered stream.
    """
    diffs = abs(tpas - ph)
    most_likely = np.argmin(diffs)
    conf = diffs[most_likely] / diffs
    if np.any(conf[conf<1] > threshold):
        return -1
    return tpas[most_likely]
get_tpas_from_stream = np.vectorize(get_tpa_from_stream, excluded=[1, 2])


class Stream_CSMPL(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'CSMPL'.
    The data is stored in `*.csmpl` files (for each channel separately). Additionally, we need a `*.par` file to read the start timestamp of the stream data from.
    """
    def __init__(
            self,
            files: List[str],
            tp_stream: Union[str, List[str]]=None,
            tp_stream_amplitudes: Union[List[float], List[List[float]]]=None,
            tp_confidence_threshold: float=0.5
            ):
        super().__init__(files=files)

        if not any([x.endswith('.par') or x.endswith('.json') for x in files]):
            raise ValueError("You have to provide either a '.par' or '.json' file to construct this class.")
        if any([x.endswith('.par') for x in files]) and any([x.endswith('.json') for x in files]):
            raise ValueError("You may only provide one of a '.par' and a '.json' file, not both.")
        if not any([x.endswith('.csmpl') for x in files]):
            raise ValueError("You have to provide at least one '.csmpl' file to construct this class.")
        if any([os.path.splitext(x)[-1] not in [".csmpl", ".par", ".json", ".test_stamps", ".dig_stamps"] for x in files]):
            raise ValueError("Only file extensions ['.csmpl', '.par'] are supported.")

        if any([tp_stream is None, tp_stream_amplitudes is None]) and \
           any([tp_stream is not None, tp_stream_amplitudes is not None]):
            raise ValueError("If either 'tp_stream' or 'tp_stream_amplitudes is defined, both must be")

        # Work with list of TP streams in case there is more than one pulser
        if isinstance(tp_stream, str):
            tp_stream = [tp_stream]
        tp_stream_amplitudes = np.array(tp_stream_amplitudes)
        if tp_stream_amplitudes.ndim == 1:
            tp_stream_amplitudes = np.tile(tp_stream_amplitudes, (len(tp_stream), 1))
        self._tp_streams = tp_stream
        self._tp_stream_amps = tp_stream_amplitudes
        self._tp_stream_conf = tp_confidence_threshold

        csmpl_paths = [x for x in files if x.endswith('.csmpl')]
        test_path = [x for x in files if x.endswith('.test_stamps')]
        dig_path = [x for x in files if x.endswith('.dig_stamps')]

        # Offset from the dig_stamps file (assuming a 10 MHz clock)
        offset = 0 if not dig_path else int(get_offset(dig_path[0])/10)

        if any([x.endswith('.par') for x in files]):
            par_path = [x for x in files if x.endswith('.par')][0]
            self._par_file = PARFile(par_path)
            self._start = int(1e6*self._par_file.start_s + self._par_file.start_us - offset)
            self._dt = self._par_file.time_base_us
            self._par_path = par_path

        elif any([x.endswith('.json') for x in files]):
            self._json_path = [x for x in files if x.endswith('.json')][0]
            with open(self._json_path, 'r') as f:
                self._config = json.load(f)

            if "start_ts" in self._config:
                # Start time is Unix timestamp
                self._start = int(1e6*self._config["start_ts"] - offset)
            elif "start_s" in self._config and "start_us" in self._config:
                # Timestamp is separated into seconds and microseconds, a la .par files
                self._start = int(1e6*self._config["start_s"] + self._config["start_us"] - offset)
            else:
                # Improperly formatted config file
                raise ValueError(
                        "Config (.json) file must contain either "
                        "\"start_ts\", or \"start_s\" and \"start_us\"."
                        )

            if "time_base_us" not in self._config:
                raise ValueError("Config (.json) file must contain \"time_base_us\".")
            self._dt = self._config["time_base_us"]

        self._data = dict()

        for fname in csmpl_paths:
            name = os.path.splitext(os.path.basename(fname))[0]
            if name.split("_")[-1].lower().startswith("ch"): name = name.split("_")[-1]
            self._data[name] = BinaryFile(path=fname, dtype=np.dtype(np.int16))

        if test_path:
            if not dig_path:
                raise Exception("When including testpulse information using a '.test_stamps' file, you also have to provide the corresponding '.dig_stamps' file.")
            test_path = test_path[0]
            test_h, tpas, test_chs = get_test_stamps(test_path)

            self._tpas = dict()
            self._tp_timestamps = dict()

            for k in list(set(test_chs)):
                mask = test_chs == k
                self._tpas[str(k)] = tpas[mask]
                # assuming 10 MHz clock
                self._tp_timestamps[str(k)] = self.start_us + np.array(test_h[mask]*3600*1e6, dtype=np.int64) + offset

        self._keys = list(self._data.keys())

    def __len__(self):
        return len(self._data[self.keys[0]])

    def __enter__(self):
        for bin_file in self._data.values(): bin_file.__enter__()
        return self

    def __exit__(self, typ, val, tb):
        for bin_file in self._data.values(): bin_file.__exit__(typ, val, tb)

    def get_trace(self, key: str, where: slice, voltage: bool = True):
       data = self._data[key][where]
       return ai.data.convert_to_V(data, bits=16, min=-10, max=10) if voltage else data

    @property
    def start_us(self):
        return self._start

    @property
    def dt_us(self):
        return self._dt

    @property
    def keys(self):
        return self._keys

    @property
    def tp_keys(self):
        if not hasattr(self, '_tpas'):
            return []
        else:
            return list(self._tpas.keys())

    @property
    def tpas(self):
        if not hasattr(self, '_tpas'):
            if self._tp_streams is None:
                raise KeyError("Testpulse amplitudes not available. Include a '.test_stamps' and a '.dig_stamps' file when constructing this class to use this feature.")
            else:
                raise KeyError("Testpulse amplitudes not available, but a stream is available for triggering TP data. Call the 'trigger_tp_stream()' method of this object to calculate it before using this feature")
        return self._tpas

    @property
    def tp_timestamps(self):
        if not hasattr(self, '_tp_timestamps'):
            if self._tp_streams is None:
                raise KeyError("Testpulse timestamps not available. Include a '.test_stamps' and a '.dig_stamps' file when constructing this class to use this feature.")
            else:
                raise KeyError("Testpulse timestamps not available, but a stream is available for triggering TP data. Call the 'trigger_tp_stream()' method of this object to calculate it before using this feature")
        return self._tp_timestamps
        return self._tp_timestamps


    def trigger_tp_stream(self, **kwargs):
        """
        Trigger a dedicated testpulse stream, and determine the TPAs from the list given in
        the constructor.

        Triggering is done using :func:`cait.versatile.trigger_zscore`, which by default
        returns a substantial number of noise triggers on TP streams; these are currently
        rejected by removing triggers with a pulse height below half the lowest TPA.

        If the pulse height of a triggered TP is consistent with more than one TPA, its
        value is set to -1.

        :param **kwargs: Keyword arguments passed to :func:`cait.versatile.trigger_zscore`.
        :type **kwargs: dict
        """
        if hasattr(self, "_tpas") > 0 or self._tp_streams is None:
            # Already done, or no data available
            return

        self._tpas = dict()
        self._tp_timestamps = dict()

        _stream = Stream_CSMPL(self._tp_streams + [self._par_path if hasattr(self, "_par_path") else self._json_path])
        # 2**15 samples should be long enough to capture TP, and we shouldn't be sending
        # them faster than this.
        _record_length = int(2**15)

        for i in range(len(self._tp_streams)):
            print(f"Triggering channel {_stream.keys[i]} for TP data...")
            inds, _ = trigger_zscore(
                    _stream[_stream.keys[i]],
                    record_length = _record_length,
                    skip_postprocessing = True,
                    **kwargs,
                    )

            mp = MainParameters(dt_us = self._dt, bcs = dict(length=1))
            it = _stream.get_event_iterator(_stream.keys, record_length = _record_length, inds = inds)

            out = np.array(apply(mp, it, pb_prefix="Calculating pulse heights"))

            # Cut out noise
            ph = out[0]
            cut = ph > self._tp_stream_amps[i].min() / 2
            ph = ph[cut]
            inds = np.array(inds)[cut]

            tpas = get_tpas_from_stream(ph, self._tp_stream_amps[i], self._tp_stream_conf)

            self._tpas[str(i)] = tpas
            self._tp_timestamps[str(i)] = _stream.time[inds]
