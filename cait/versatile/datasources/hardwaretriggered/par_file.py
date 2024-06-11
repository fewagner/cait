import re
from typing import Union

from ....readers import TextFile

CHANNEL_NAME_FORMATS = [r"ch(\d+): (.+)", r"ch(\d+), (.+)"]

# Would be cool to input parameters manually in case we don't have a PAR file. 
# This should go directly into this class so that higher level classes don't have to handle it.
class PARFile:
    def __init__(self, arg: Union[str, dict]):
        # if isinstance(arg, str):
        if not arg.endswith(".par"):
            raise ValueError("Unrecognized file extension. Please input a *.par file.")
        
        with TextFile(arg) as f:
            s = f.read()

        match = re.findall(r"Timeofday at start\s*\[s\].*Timeofday at start\s*\[us\].*Timeofday at stop\s*\[s\].*Timeofday at stop\s*\[us\].*Measuring time\s*\[h\].*Integers in header.*Unsigned longs in header.*Reals in header.*DVM channels.*Record length.*Time base\s*\[us\]", s, re.DOTALL)
        if not match:
            raise ValueError("Unable to extract data from file.")
        
        self.s = s
        # elif isinstance(arg, dict):
        #     if not all([['start_s', 
        #                  'start_us', 
        #                  'stop_s', 
        #                  'stop_us', 
        #                  'measuring_time_h', 
        #                  'ints_in_header', 
        #                  'uslongs_in_header',
        #                  'reals_in_header',
        #                  'dvm_channels',
        #                  'record_length',
        #                  'records_written',
        #                  'time_base_us']])
        # else:
        #     raise NotImplementedError(f"Unrecognized input type '{type(arg)}'.")

    def __repr__(self):
        d = {k: getattr(self, k) for k in ["start_s", "measuring_time_h", "ints_in_header", "dvm_channels", 
                                           "record_length", "records_written", "time_base_us"]}
        return f'{self.__class__.__name__}({d})'
    
    @property
    def start_s(self):
        return int(re.findall(r"Timeofday at start\s*\[s\]\s*:\s+(\d+)", self.s)[0])

    @property
    def start_us(self):
        return int(re.findall(r"Timeofday at start\s*\[us\]\s*:\s+(\d+)", self.s)[0])

    @property
    def stop_s(self):
        return int(re.findall(r"Timeofday at stop\s*\[s\]\s*:\s+(\d+)", self.s)[0])

    @property
    def stop_us(self):
        return int(re.findall(r"Timeofday at stop\s*\[us\]\s*:\s+(\d+)", self.s)[0])
    
    @property
    def measuring_time_h(self):
        return float(re.findall(r"Measuring time\s*\[h\]\s*:\s+([0-9]*[.]?[0-9]+)", self.s)[0])

    @property
    def ints_in_header(self):
        return int(re.findall(r"Integers in header\s*:\s+(\d+)", self.s)[0])

    @property
    def uslongs_in_header(self):
        return int(re.findall(r"Unsigned longs in header\s*:\s+(\d+)", self.s)[0])

    @property
    def reals_in_header(self):
        return int(re.findall(r"Reals in header\s*:\s+(\d+)", self.s)[0])

    @property
    def dvm_channels(self):
        return int(re.findall(r"DVM channels\s*:\s+(\d+)", self.s)[0])

    @property
    def record_length(self):
        return int(re.findall(r"Record length\s*:\s+(\d+)", self.s)[0])

    @property
    def records_written(self):
        return int(re.findall(r"Records written\s*:\s+(\d+)", self.s)[0])

    @property
    def time_base_us(self):
        return int(re.findall(r"Time base\s*\[us\]\s*:\s+(\d+)", self.s)[0])

    @property
    def has_channel_names(self):
        return bool(self.channel_names)
    
    @property
    def channel_names(self):
        for fmt in CHANNEL_NAME_FORMATS:
            out = {int(i[0])-1: i[1].strip() for i in re.findall(fmt, self.s)}
            if out: return out
            
        return {}