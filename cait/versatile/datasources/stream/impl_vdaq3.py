import json
from typing import List, Union

import numpy as np

import cait as ai

from ....readers import BinaryFile
from .impl_vdaq2 import VDAQ2_TP_TS, VDAQ2_TPAS
from .streambase import StreamBaseClass

UUID_SIZE = 37 # bytes (including the \x00 separator)
UUID_SINGLE_CH = "b0de7478-e909-4c2b-9415-e14792e0510d"
UUID_SINGLE_CH_WITH_TRAILER = "010cfb4a-71ae-4185-ab44-168c108623e0"

UINT16_SIZE = np.dtype(np.uint16).itemsize
UINT64_SIZE = np.dtype(np.uint64).itemsize


class Stream_VDAQ3(StreamBaseClass):
    """
    Implementation of StreamBaseClass for hardware 'vdaq3'.
    VDAQ3 data is stored in .dat files. Its header contains instructions on how to read the data and all recorded channels are stored in a separate file.

    There are two variations of the files: One includes information about the testpulses/controlpulses. In this case, the TPAs and testpulse timestamps are readily available. Another format does **not** include testpulse information. In this case, you have to enter the TP channel also in the list of ``files``. It is then treated like the VDAQ2 variant, where the TP channels are triggered to obtain the TPAs and TP timestamps. To modify the trigger parameters, change the inputs ``dac_trig_thr`` and ``dac_trig_win_len_ms``.

    :param files: The binary files with the stream data (including the file extension).
    :type files: List[str]
    :param dac_trig_thr: **Only for VDAQ2-style TP-channel trigger:** Trigger threshold (in sigmas) to find testpulses from the DAC channels. Defaults to 5 sigmas
    :type dac_trig_thr: float, optional
    :param dac_trig_win_len_ms: **Only for VDAQ2-style TP-channel trigger:** Trigger window length (in ms) to find testpulses from the DAC channels. Defaults to 500 ms
    :type dac_trig_win_len_ms: int, optional
    """
    def __init__(self, 
                 files: Union[str, List[str]],
                 dac_trig_thr: float = 5.,      # sigmas
                 dac_trig_win_len_ms: int = 500 # ms
                 ):
        super().__init__(files=files, dac_trig_thr=dac_trig_thr, dac_trig_win_len_ms=dac_trig_win_len_ms)
        
        if type(files) is str: files = [files]

        starts, dTs, lengths, uuids, precs = [], [], [], [], []
        self._data, self._header, self._trailer = dict(), dict(), dict()
        self._tp_timestamps, self._tpas = dict(), dict()
        
        # Data is 24 bits, i.e. 3 bytes long. Read 3 bytes at a time
        # Possibly also 32 bit
        # This is decided by 'sample_size_bytes' in the header which is either 3 or 4
        data_dtype = {
            3: np.dtype([('byte1', '<u1'), 
                         ('byte2', '<u1'), 
                         ('byte3', '<u1')]),
            4: np.dtype('<u4')
        }

        # The data format is documented here: https://cryocluster-vccs.docs.cern.ch/data-formats/single-channel-file/
        for f in files:
            # Starting from the second byte, we extract the UUID which defines the file format
            uuid = BinaryFile(path=f, dtype=np.dtype(f"V{UUID_SIZE}"), count=1, offset=2)[0].tobytes().decode("ascii").split("\x00")[0]

            if uuid not in [UUID_SINGLE_CH, UUID_SINGLE_CH_WITH_TRAILER]:
                raise NotImplementedError(f"uuid '{uuid}' is an unsupported file format for this stream object. Valid uuids: {[UUID_SINGLE_CH, UUID_SINGLE_CH_WITH_TRAILER]}")

            uuids.append(uuid)
            size_file = len(BinaryFile(path=f, dtype=np.dtype(np.int8)))

            if uuid == UUID_SINGLE_CH:
                # The first two bytes give the offset to the sample data
                offset_samples = BinaryFile(path=f, dtype=np.dtype(np.int16), count=1)[0]
                offset_json = UINT16_SIZE + UUID_SIZE
                size_json = offset_samples - offset_json
                trailer = dict()
                # This format doesn't save any additional testpulse information
                tp_ts = list()
                tpas = list()

                # These attributes are needed for triggering the TP channels
                self._dac_trig_thr = dac_trig_thr
                self._dac_trig_win_len_ms = dac_trig_win_len_ms
                self._dac_trig_win_len = int(1000*dac_trig_win_len_ms/self._dt)
                
            elif uuid == UUID_SINGLE_CH_WITH_TRAILER:
                # The first 8 bytes after the uuid give the offset to the sample data
                # The second 8 bytes after the uuid give the offset to the trailer
                offset_samples, offset_trailer = BinaryFile(path=f, dtype=np.dtype(np.uint64), count=2, offset=UINT16_SIZE+UUID_SIZE)
                offset_json = UINT16_SIZE + UUID_SIZE + 2*UINT64_SIZE
                size_json = offset_samples - offset_json
                size_trailer = size_file - offset_trailer
                trailer = json.loads(
                    BinaryFile(
                        path=f, 
                        dtype=np.dtype(f"V{size_trailer}"), 
                        offset=offset_trailer, 
                        count=1
                    )[0].tobytes().decode("ascii").split("\x00")[0]
                )

                # This format stores testpulse/controlpulse information in the trailer
                tp_ts = list()
                tpas = list()
                for d in trailer["trigger_parameter"]:
                    if "injected_pulse" in d.keys():
                        ip = d["injected_pulse"]
                        if all([x in ip.keys() for x in ["energy", "ts_start_ns", "type"]]):
                            if ip["type"] in ["test_pulse", "control_pulse"]:
                                # Don't forget to convert ns to us
                                tp_ts.append(ip["ts_start_ns"]//1000)
                                tpas.append(ip["energy"])

            size_data = size_file - size_trailer - offset_samples

            header = json.loads(
                BinaryFile(
                    path=f, 
                    dtype=np.dtype(f"V{size_json}"), 
                    offset=offset_json, 
                    count=1
                )[0].tobytes().decode("ascii").split("\x00")[0]
            )
            
            dTs.append(header["timestep_ns"])
            
            # There is also a 'ts_utc_ns' which gives the UTC time but it is not
            # synchronized with e.g. muon veto
            starts.append(header["ts_ns"])

            channel_name = header["channel_id"]
            prec = header["sample_size_bytes"]
            
            if prec not in [3, 4]:
                raise NotImplementedError(f"Only 3 and 4 byte data precision are supported. Got {prec}.")
                
            precs.append(prec)

            self._data[f"Ch{channel_name}"] = BinaryFile(path=f, 
                                                         dtype=data_dtype[prec], 
                                                         count=size_data//prec,
                                                         offset=offset_samples)
            
            if uuid == UUID_SINGLE_CH_WITH_TRAILER and len(tp_ts)>0:
                self._tp_timestamps[f"TP_Ch{channel_name}"] = tp_ts
                self._tpas[f"TP_Ch{channel_name}"] = tpas

            # HERE WE COULD PROBABLY USE THE INFO IN THE HEADER AT SOME POINT
            lengths.append(len(self._data[f"Ch{channel_name}"]))

            # Save header and trailer
            self._header[f"Ch{channel_name}"] = header
            self._trailer[f"Ch{channel_name}"] = trailer
        
        if len(np.unique(starts)) > 1:
            raise ValueError(f'Files have to start at the same time to be treated together. Got {starts}.')
        if len(np.unique(dTs)) > 1:
            raise ValueError(f'Files have to have the same time-delta to be treated together. Got {dTs}.')
        if len(np.unique(lengths)) > 1:
            raise ValueError(f'Files have to have the same length to be treated together. Got {lengths}.')
        if len(np.unique(uuids)) > 1:
            raise ValueError(f'Files have to have the same uuid string to be treated together. Got {uuids}.')
        if len(np.unique(precs)) > 1:
            raise ValueError(f'Files have to be recorded with identical byte precision to be treated together. Got {precs}.')
        
        # Number of data points in stream
        self._len = lengths[0]
        # Start timestamp of the file in us (header value is in ns)
        self._start = int(starts[0]//1000)
        # Temporal step size in us (= inverse sampling frequency)
        self._dt = dTs[0]//1000
        # The byte precision
        self._prec = precs[0]
        # The file type
        self._uuid = uuids[0]
        
    def __len__(self):
        return self._len
    
    def __enter__(self):
        for bin_file in self._data.values(): bin_file.__enter__()
        return self
    
    def __exit__(self, typ, val, tb):
        for bin_file in self._data.values(): bin_file.__exit__(typ, val, tb)
    
    def get_trace(self, key: str, where: slice, voltage: bool = True):
        data = self._data[key][where]
 
        if self._prec == 3:
            # If written as 24bit values, here, we convert them to 32 bits such that numpy can handle them
            adc_32bit = np.vstack([
                    np.zeros_like(data["byte1"]),
                    data["byte1"], 
                    data["byte2"], 
                    data["byte3"]
                ]).flatten("F").view("<u4")

            return ai.data.convert_to_V(adc_32bit, bits=32, min=-20, max=20) if voltage else adc_32bit
        
        elif self._prec == 4:
            
            result = np.zeros_like(data)
    
            for i, val in enumerate(data):
                bin_val = format(val, '032b')  # '032b' ensures a 32-bit representation
                rearranged_bin = bin_val[-24:] + bin_val[:8] # Rearrange the bits: last 24 bits come first 
                result[i] = int(rearranged_bin, 2)
    
                
            adc_32bit = result 
          
            return ai.data.convert_to_V(adc_32bit, bits=32, min=-20, max=20) if voltage else adc_32bit
    
    @property
    def start_us(self):
        return self._start
    
    @property
    def dt_us(self):
        return self._dt
    
    @property
    def keys(self):
        return list(self._data.keys())
    
    @property
    def tp_keys(self): 
        if self._uuid == UUID_SINGLE_CH:
            # Allow all 'regular' channels to be testpulse channels (like in VDAQ2)
            return self.keys
        elif self._uuid == UUID_SINGLE_CH_WITH_TRAILER:
            # Return TP information from trailer
            return list(self._tp_timestamps.keys())
        else:
            return dict()

    @property
    def tpas(self):
        if self._uuid == UUID_SINGLE_CH:
            return VDAQ2_TPAS(self)
        elif self._uuid == UUID_SINGLE_CH_WITH_TRAILER:
            return self._tpas
        else:
            return dict()

    @property
    def tp_timestamps(self):
        if self._uuid == UUID_SINGLE_CH:
            return VDAQ2_TP_TS(self)
        elif self._uuid == UUID_SINGLE_CH_WITH_TRAILER:
            return self._tp_timestamps
        else:
            return dict()