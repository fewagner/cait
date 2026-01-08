import copy
from functools import partial
from math import exp
from typing import List

import numba
import numpy as np
import scipy as sp

from .streambase import StreamBaseClass


##################################
############ HELPER ##############
##################################
@numba.njit
def _pulse(t, t0, An, At, tau_n, tau_in, tau_t):
    """Numba accelerated _pulse shape evaluation for scalar."""
    t_ = t - t0
    # Before onset
    if t_ <= 0: 
        return 0.0
    # Already fallen off sufficiently
    if t_ > 10*tau_t:
        return 0.0
    
    return An * ( exp(-t_/tau_n) - exp(-t_/tau_in) ) + At * ( exp(-t_/tau_t) - exp(-t_/tau_n) )

@numba.vectorize([numba.float64(*(7*[numba.float64]))])
def _pulse_vec(t, t0, An, At, tau_n, tau_in, tau_t):
    """Numba accelerated _pulse shape evaluation for array-input."""
    return _pulse(t, t0, An, At, tau_n, tau_in, tau_t)

@numba.guvectorize([
    (numba.float64[:], numba.float64[:], numba.float64[:], *(5*[numba.float64]), numba.float64[:])
    ], 
    '(n),(m),(m),(),(),(),(),()->(n)',
    target="parallel",
)
def _pulse_sum(t, ts, phs, An, At, tau_n, tau_in, tau_t, res):
    """Numba accelerated _pulse shape evaluation for array-input and multiple _pulses placed at 'ts' with _pulse heights 'phs'."""
    res[:] = 0.0
    relevant = (ts <= t[-1])*(ts > t[0] - 10*tau_t)
    for i in numba.prange(len(ts)):
        if relevant[i]:
            res[:] += phs[i]*_pulse_vec(t, ts[i], An, At, tau_n, tau_in, tau_t)

def gen_noise(sl: slice, len_stream: int, base_seed: int, scale: float, chunk_size: int = 100000):
    """Generate random but reproducible noise for a part 'sl' of a stream with length 'len_stream'."""
    recovery_sl = slice(None, None, None)
    if not isinstance(sl, slice):
        if not isinstance(sl, np.ndarray):
            raise TypeError(f"Unsupported type {type(sl)} for input 'sl'.")
        recovery_sl = sl - np.min(sl)
        sl = slice(np.min(sl), np.max(sl)+1)
        
    # Sanitize input
    start = 0 if not sl.start else sl.start
    end = len_stream - 1 if not sl.stop else sl.stop
    step = 1 if not sl.step else sl.step
    
    # Allow negative indexing ([-1], [-10:], etc.)
    if start<0:
        if end==len_stream-1 and start==-1:
            end = len_stream
        start += len_stream
    if end<0:
        end += len_stream
    
    # Extend slice to go from a multiple of chunk_size
    # to another multiple of chunk_size
    temp_start = (start//chunk_size)*chunk_size
    temp_end = (end//chunk_size + 1)*chunk_size
    recover_start = start%chunk_size
    recover_end = chunk_size - end%chunk_size
    
    n_chunks = (temp_end - temp_start)//chunk_size
    i_first_chunk = temp_start//chunk_size
    
    out = np.zeros(temp_end-temp_start, dtype=np.float64)
    
    # Generate random numbers with the seed fixed to the start of the chunk.
    # (that's why it's important to extend the chunks above)
    for i in range(n_chunks):
        out[i*chunk_size:(i+1)*chunk_size] = sp.stats.norm.rvs(
            loc=0, scale=scale, size=chunk_size, random_state=(i+i_first_chunk)*base_seed
        )
    
    return out[recover_start:-recover_end:step][recovery_sl]

def _saturate(y, plateau=0.3):
    """Imitate the effect of saturation close to the plateau of the transition."""
    out = np.zeros_like(y)
    positive_flag = y > 0
    
    out[~positive_flag] = y[~positive_flag]
    # Apply tanh such that the plateau sits at 'plateau'
    out[positive_flag] = np.tanh(y[positive_flag]/plateau)*plateau
    
    return out  

def _default_spectrum_cdf(x, x_max: float = 40):
    """Default spectrum consisting of a constant background and an iron line."""
    w59, w64, wc = 1, 0.2, 0.5
    return (
        w59*sp.stats.norm.cdf(x, loc=5.9, scale=0.01) 
        + w64*sp.stats.norm.cdf(x, loc=6.4, scale=0.01)
        + wc*sp.stats.uniform.cdf(x, loc=0, scale=x_max)
    )/(w59 + w64 + wc)

##################################
####### CLASS DEFINITION #########
##################################

class MockStream(StreamBaseClass):
    """
    Mock implementation of a stream. Can be used for tutorials and testing features. 

    :param duration_h: The length of the stream in hours. Defaults to 1 hour.
    :type duration_h: float, optional
    :param dt_us: The microsecond timebase of the stream. Defaults to 10 us.
    :type dt_us: float, optional
    :param pulse_shape: The amplitudes and time constants of the pulses to simulate. Parameters are ``(An, At, tau_n, tau_in, tau_t)``. One list for each channel (number of channels simulated is determined by the number of lists in this argument). 
    :type pulse_shape: List[List[float]], optional
    :param rate_Hz: The rate (in Hz) at which events are simulated. Defaults to 0.1 Hz.
    :type rate_Hz: float, optional
    :param tpa: The TPA pattern that repeats throughout the stream.
    :type tpa: List[float], optional
    :param tp_shape: The TPA pulse parameters for each channel (see ``pulse_shape``). Number of channels has to match number of channels in 'pulse_shape'.
    :type tp_shape: List[List[float]], optional
    :param tp_interval_s: The number of seconds between consecutive testpulses (whose amplitude is defined by ``tpa``). Defaults to 5 seconds.
    :type tp_interval_s: float, optional
    :param spectrum_cdf: The (keV) spectrum (CDF) for the data to simulate. Defaults to a spectrum consisting of an Fe55 double peak and a constant background up to ``max_energy_keV`` (see below).
    :type spectrum_cdf: callable, optional
    :param baseline_sig: The baseline standard deviation (in V) for each channel.
    :type baseline_sig: List[float], optional
    :param max_energy_keV: The maximum energy to simulate. Defaults to 40 keV
    :type max_energy_keV: float, optional
    :param seed: When set, the random numbers generated in the simulation become reproducible. Defaults to None, i.e. new random numbers each time an object is instantiated.
    :type seed: int, optional

    .. code-block:: python

        import cait.versatile as vai

        stream = vai.MockStream(seed=137)
        vai.StreamViewer(stream)
    """
    _auto_gen_config = {
        "chunk_size": 100000,
        "bl_offset": -8.3,
        "saturation_plateau_keV": 30.0,
    }
    def __init__(self, 
                 duration_h: float = 1.,
                 dt_us: int = 10,
                 pulse_shape: List[List[float]] = [[0.5, 0.5, 0.3, 0.1, 10.0], [0.3, 0.5, 0.3, 0.01, 4.0]],
                 rate_Hz: float = 0.1,
                 tpa: List[float] = [1., 2., 3., 4., 100.],
                 tp_shape: List[List[float]] = [[0.5, 0.5, 0.3, 0.01, 20.], [0.5, 0.5, 0.3, 0.01, 10.]],
                 tp_interval_s: float = 5.,   
                 spectrum_cdf: callable = None,
                 baseline_sig: List[float] = [0.005, 0.001],
                 max_energy_keV: float = 40.,
                 seed: int = None,
                ):
        super().__init__(
            duration_h=duration_h, 
            dt_us=dt_us,
            pulse_shape=pulse_shape,
            rate_Hz=rate_Hz,
            tpa=tpa,
            tp_shape=tp_shape,
            tp_interval_s=tp_interval_s,
            spectrum_cdf=spectrum_cdf,
            baseline_sig=baseline_sig,
            max_energy_keV=max_energy_keV,
            seed=seed,
            )
        
        if len(pulse_shape) != len(tp_shape):
            raise ValueError(f"'pulse_shape' and 'tp_shape' need to have the same number of channels.")
        if len(pulse_shape) != len(baseline_sig):
            raise ValueError(f"'pulse_shape' and 'baseline_sig' need to have the same number of channels.")
        for ps in pulse_shape + tp_shape:
            if len(ps) != 5:
                raise ValueError("All pulse shapes must have 5 parameters (An, At, tau_n, tau_in, tau_t).")
        
        self._start = 1426321613000000
        self._dt_us = dt_us
        self._len = int((duration_h*3600*1e6)//dt_us)

        self._n_ch = len(pulse_shape)
        
        # The factor that relates energy and pulse heights
        cpe = 160
        # Slight modification for testpulses
        cpe_tp_mod = 1/10
        # Slight modification for additional channels
        cpe_add = [cpe*10*k for k in range(1, self._n_ch)]

        
        if spectrum_cdf is None:
            spectrum_cdf = partial(_default_spectrum_cdf, x_max=max_energy_keV)
            
        if seed is None:
            seed = np.random.randint(0, 10000)
            
        # Invert CDF and draw random numbers
        x = np.linspace(0, max_energy_keV, 10000)
        y = spectrum_cdf(x)
        poly = sp.interpolate.PchipInterpolator(y, x)
        phs = poly(sp.stats.uniform.rvs(size=int(duration_h*3600*rate_Hz), random_state=seed))
        self._phs = {f"Ch{k}": phs/c  for k, c in enumerate([cpe] + cpe_add)}
        
        # Draw equally many event timestamps
        ts = self.time[np.sort(sp.stats.randint.rvs(0, self._len, size=len(phs), random_state=2*seed))]
        self._ts = {f"Ch{k}": ts-5000*k for k in range(self._n_ch)}
        
        # Normalise pulse shape to 1 
        new_pulse_shape = copy.deepcopy(pulse_shape)
        for i in range(len(new_pulse_shape)):
            scale = np.max(_pulse_vec(np.linspace(-10, 1000, 100000), 0, *new_pulse_shape[i]))
            new_pulse_shape[i][0] /= scale
            new_pulse_shape[i][1] /= scale
        self._pulse_shape =  {f"Ch{k}": ps for k, ps in enumerate(new_pulse_shape)}
        
        # Set up testpulses
        n_tp = int(duration_h*3600/(tp_interval_s)) - 1
        self._tp_ts = self._start + np.arange(1, n_tp+1)*int(tp_interval_s*1e6)
        self._tpas = np.tile(tpa, n_tp//len(tpa) + 1)[:n_tp]
        self._tp_phs = {f"TP{k}": self._tpas/(c*cpe_tp_mod)  for k, c in enumerate([cpe] + cpe_add)}
        
        new_tp_shape = copy.deepcopy(tp_shape)
        for i in range(len(new_tp_shape)):
            scale = np.max(_pulse_vec(np.linspace(-10, 1000, 100000), 0, *new_tp_shape[i]))
            new_tp_shape[i][0] /= scale
            new_tp_shape[i][1] /= scale
        self._tp_shape = {f"TP{k}": ps for k, ps in enumerate(new_tp_shape)}
        
        self._bl_sig = {f"Ch{k}": bs for k, bs in enumerate(baseline_sig)}
        self._seed = 137*seed
        
        self._plateau =  {f"Ch{k}": self._auto_gen_config["saturation_plateau_keV"]/c for k, c in enumerate([cpe] + cpe_add)}
        
    def __enter__(self):
        return self
    
    def __exit__(self, typ, val, tb):
        ...
        
    def __len__(self):
        return self._len
    
    def get_trace(self, key: str, where: slice, voltage: bool = True):
        bl = gen_noise(
            where, 
            len(self), 
            base_seed=self._seed, 
            scale=self._bl_sig[key], 
            chunk_size=self._auto_gen_config["chunk_size"],
        )
        events = _pulse_sum(
            np.array(self.time[where]/1000, dtype=np.float64), 
            np.array(self._ts[key]/1000, dtype=np.float64), 
            np.array(self._phs[key], dtype=np.float64), 
            *self._pulse_shape[key]
        )
        tps = _pulse_sum(
            np.array(self.time[where]/1000, dtype=np.float64), 
            np.array(self._tp_ts/1000, dtype=np.float64), 
            np.array(self._tp_phs[f"TP{key[-1]}"], dtype=np.float64), 
            *self._tp_shape[f"TP{key[-1]}"]
        )
        
        return self._auto_gen_config["bl_offset"] + _saturate(bl + events + tps, plateau=self._plateau[key])
        
    @property
    def start_us(self):
        return self._start
    
    @property
    def dt_us(self):
        return self._dt_us
    
    @property
    def keys(self):
        return [f"Ch{k}" for k in range(self._n_ch)]
    
    @property
    def tp_keys(self):
        return [f"TP{k}" for k in range(self._n_ch)]
    
    @property
    def tpas(self):
        return {f"TP{k}": self._tpas for k in range(self._n_ch)}
        
    @property
    def tp_timestamps(self):
        return {f"TP{k}": self._tp_ts for k in range(self._n_ch)}