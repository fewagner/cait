import json
import os
from typing import List, Union, Dict
from functools import partial

import numpy as np

import cait as ai

from ....readers import BinaryFile
from ..hardwaretriggered.par_file import PARFile
from .impl_csmpl import Stream_CSMPL
from ...functions import apply
from ...functions.trigger.trigger_zscore import zscore_chunk
from ...functions.trigger.triggerbase import trigger_base
from ...eventfunctions import RemoveBaseline


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


def _max_and_argmax(x):
    ind = np.argmax(x)
    return ind, x[ind]


class Stream_TPStream(Stream_CSMPL):
    """
    Implementation of StreamBaseClass for hardware 'CSMPL', with testpulse data recorded to a stream.
    The data is stored in `*.csmpl` files (for each channel separately). Additionally, we need a `*.par` file to read the start timestamp of the stream data from.

    This class inherits from :class:`cait.versatile.datasources.stream.impl_csmpl.Stream_CSMPL`,
    and so supports all of its features and requirements.  In addition, it
    enables the triggering of a separate `.csmpl` stream to which raw
    testpulses have been recorded, and exposes those timestamps/amplitudes
    when required (e.g. for rejecting testpulses during triggering of the
    real data).  Triggering of the TP stream is done on the fly, i.e. when any
    of the TP data is first accessed (timestamps/indices, tpas, etc.).
    """
    def __init__(
            self,
            files: List[str],
            tp_stream_file = None,
            unique_tpas: Union[List[float], List[List[float]]] = None,
            daq_file: str = None,
            trigger_threshold: float = 10,
            confidence_threshold: float = 0.5,
            record_length = 2**15,
            #tp_stream_data: Union[str, List[str], Dict] = {},
            ):
        #if isinstance(tp_stream_data, (str, list)):
        #    tp_stream_data = {"tp_path": tp_stream_data}

        self._tp_stream_file = tp_stream_file

        self._trigger_threshold = trigger_threshold
        self._confidence_threshold = confidence_threshold
        self._record_length = record_length

        if unique_tpas is not None:
            self._unique_tpas = unique_tpas
        elif daq_file is not None:
            with open(daq_file) as f:
                daq = json.load(f)
            self._unique_tpas = np.array(
                    daq["TestPulses"]["tp_params"]["tp_amplitudes"] +
                    [daq["TestPulses"]["tp_params"]["cp_amplitude"]]
                    )

        super().__init__(files=files)

        if hasattr(self, "tpas"):
            del self.tpas
        if hasattr(self, "tp_timestamps"):
            del self.tp_timestamps

        self._tp_stream_data = dict(
            tp_path = tp_stream_file,
            tpas = self._unique_tpas,
            trigger_threshold = trigger_threshold,
            confidence_threshold = confidence_threshold,
            record_length = record_length,
            )





    @property
    def tp_keys(self):
        if not hasattr(self, '_tpas'):
            if self._tp_stream_data is not None:
                return CSMPL_TPAS(self, **self._tp_stream_data).keys
            return []
        else:
            return list(self._tpas.keys())

    @property
    def tpas(self):
        if not hasattr(self, '_tpas'):
            if self._tp_stream_data is None:
                raise KeyError("Testpulse amplitudes not available. Include a '.test_stamps' and a '.dig_stamps' file when constructing this class to use this feature.")
            else:
                return CSMPL_TPAS(self, **self._tp_stream_data)
        return self._tpas

    @property
    def tp_timestamps(self):
        if not hasattr(self, '_tp_timestamps'):
            if self._tp_stream_data is None:
                raise KeyError("Testpulse timestamps not available. Include a '.test_stamps' and a '.dig_stamps' file when constructing this class to use this feature.")
            else:
                return CSMPL_TP_TS(self, **self._tp_stream_data)
        return self._tp_timestamps





class CSMPL_TP_Helper:
    """
    Helper class for triggering a testpulse stream.  Parent of
    :class:`cait.versatile.datasources.stream.impl_csmpl.CSMPL_TPAS` and
    :class:`cait.versatile.datasources.stream.impl_csmpl.CSMPL_TP_TS`, which
    access the actual data.

    :param stream: Stream containing data channels (NOT the TP stream).
        Generally the object which creates this class.
    :type stream: Stream_CSMPL

    :param tp_path: Path(s) to a binary file (`.csmpl` or `.bin`) containing
        the stream data for raw testpulses.
    :type tp_path: str or list of str

    :param tpas: List of unique testpulse amplitudes.  If provided, each
        triggered pulse is associated with the closest TPA. If the ratio of
        (pulse_height - closest_tpa) / (pulse_height - other_tpa) is greater
        than `confidence_threshold` for any other TPA, the TPA is considered
        inconclusive and the value is set to -1. Default: None
    :type tpas: list of float

    :param trigger_threshold: Threshold for :func:`cait.versatile.trigger_zscore`,
        in standard deviations. Default: 5
    :type trigger_threshold: float

    :param confidence_threshold: Threshold for deciding if a TP is correlated
        with a specific TPA, or if it is inconclusive. Default: 0.5
    :type confidence_threshold: float

    :param record_length: Window length for triggering, in samples. Default:
        16384
    :type record_length: intdetermining 
    """
    def __init__(
            self,
            stream: Stream_TPStream,
            tp_path: Union[str, List[str]] = None,
            tpas: Union[List[float], List[List[float]]] = None,
            trigger_threshold: float = 10,
            confidence_threshold: float = 0.5,
            record_length = 2**15,
            ):
        if tp_path is None:
            raise ValueError("This clas requires at least one path to a stream containing testpulses.")

        self._stream = stream

        if isinstance(tp_path, str):
            tp_path = [tp_path]
        self._tp_stream = Stream_CSMPL(tp_path + [stream._par_path])
        self._tpas = np.atleast_2d(np.array(tpas)) if tpas is not None else None
        self._thresh = np.atleast_1d(trigger_threshold)
        self._conf = np.atleast_1d(confidence_threshold)
        self._rl = int(record_length)

        self._tpa_fnc = np.vectorize(self.get_tpa_from_stream, excluded=[1, 2], signature='()->()')


    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


    def _trigger(self, key: str):
        if key not in self._tp_stream.keys:
            raise KeyError(f"No TP key {key} in key list: {self._tp_stream.keys}")

        if not hasattr(self._stream, "_tpas"):
            self._stream._tpas = {}
            self._stream._tp_timestamps = {}

        if key not in self._stream._tpas:
            tpas, timestamps = self.trigger_tp_stream(key)

            self._stream.tpas[key] = tpas
            self._stream.tp_timestamps[key] = timestamps


    @property
    def keys(self):
        return self._tp_stream.keys


    def trigger_tp_stream(self, key):
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
        kindex = self.keys.index(key)

        print(f"Triggering channel {key} for TP data...")
        inds, _ = trigger_base(
                stream=self._tp_stream[key],
                threshold = self._thresh[kindex],
                filter_fnc = partial(zscore_chunk, record_length = self._rl),
                record_length = self._rl,
                )

        # Calculate amplitude
        it = self._tp_stream.get_event_iterator(key, record_length=self._rl, inds=inds).with_processing(RemoveBaseline())

        argmaxs, ph = apply(_max_and_argmax, it, pb_prefix="Calculating pulse heights")
        inds = inds + argmaxs - self._rl // 4

        # Ensure uniqueness (for some reason there are doubles sometimes)
        inds, _idx = np.unique(inds, return_index=True)
        ph = ph[_idx]

        # Cut out noise and associate TP events with the TPAs sent, if available
        if self._tpas is not None:
            cut = ph > self._tpas.min() / 2
            ph = ph[cut]
            inds = np.array(inds)[cut]

            tpas = self._tpa_fnc(ph, self._tpas[kindex], self._conf)

        else:
            # If TPAs are not available, just return the pulse heights
            tpas = ph

        return tpas, self._stream.time[inds]


    @staticmethod
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
            return -1.
        return tpas[most_likely]


class CSMPL_TPAS(CSMPL_TP_Helper):
    def __getitem__(self, key: str):
        self._trigger(key)

        return self._stream._tpas[key]


class CSMPL_TP_TS(CSMPL_TP_Helper):
    def __getitem__(self, key: str):
        self._trigger(key)

        return self._stream._tp_timestamps[key]
