from typing import Union, Type, Any
from abc import ABC, abstractmethod

import numpy as np

from .analysis import apply
from .iterators import IteratorBaseClass
from .functions import OptimumFiltering
from .plot import Line

class ArrayWithBenefits(ABC, np.lib.mixins.NDArrayOperatorsMixin):
    def __repr__(self):
        return f"{self.__class__.__name__}({self._array})"
    
    def __len__(self):
        return len(self._array)
    
    def __array__(self, dtype=None):
        return self._array
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        args = (a._array if isinstance(a, self.__class__) else a for a in args)
        
        if method == "__call__":
            out = self.__class__()
            out._array = f[method](*args, **kwargs)
        else:
            out = f[method](*args, **kwargs)
        
        return out
    
    def __getitem__(self, val):
        out = self.__class__()
        out._array = self._array.__getitem__(val)
        return out
    
    @property
    def shape(self):
        return self._array.shape
    
    @property
    def ndim(self):
        return self._array.ndim
    
    @property
    @abstractmethod
    def _array(self):
        ...

    @_array.setter
    @abstractmethod
    def _array(self, array):
        ...

    @property
    @abstractmethod
    def _n_channels(self):
        ...

class SEV(ArrayWithBenefits):
    def __init__(self, data: Union[np.array, Type[IteratorBaseClass]] = None):
        if data is None:
            self._sev = np.empty(0)
            self._n_ch = 0
        elif isinstance(data, IteratorBaseClass):
            mean_pulse = np.mean(apply(lambda x: x, data), axis=0)
            # Normalize
            maxima = np.max(mean_pulse, axis=-1)
            # Cast maxima into a column vector such that vectorization works
            if maxima.ndim > 0: maxima = maxima[:, None]

            self._sev = mean_pulse/maxima
            self._n_ch = data.n_channels
        elif isinstance(data, np.ndarray):
            self._sev = data
            self._n_ch = 1 if data.ndim < 2 else data.shape[0]
    
    def from_dh(self, dh, group="stdevent", dataset="event"):
        self._sev = dh.get(group, dataset)
        self._n_ch = 1 if self._sev.ndim < 2 else self._sev.shape[0]
        return self
        
    def to_dh(self, dh, group="stdevent", dataset="event", **kwargs):
        dh.set(group, **{dataset: self._sev}, **kwargs)
        
    def from_file(self, file: str):
        ...
        
    def to_file(self):
        ...

    def show(self, dt: int = None, **kwargs):
        if dt is not None:
            if 'x' in kwargs.keys():
                print("Setting data for the x-axis overrides input 'dt'.")
            else:
                # TODO: convert to proper time
                kwargs['x'] = None

        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(self._array):
                y[f'channel {i}'] = channel
        else:
            y = self._array

        return Line(y, **kwargs)

    @property
    def _array(self):
        return self._sev
    
    @_array.setter
    def _array(self, array):
        if self._sev.size == 0:
            self._sev = array
            self._n_ch = 1 if array.ndim < 2 else array.shape[0]
        else:
            raise Exception("SEV._array can only be set as long as it is empty.")
    
    @property
    def _n_channels(self):
        return self._n_ch

class NPS(ArrayWithBenefits):
    def __init__(self, data: Union[np.array, Type[IteratorBaseClass]] = None):
        if data is None:
            self._nps = np.empty(0)
            self._n_ch = 0
        elif isinstance(data, IteratorBaseClass):
            self._nps = np.mean(apply(lambda x: np.abs(np.fft.rfft(x))**2, data), axis=0)
            self._n_ch = data.n_channels
        elif isinstance(data, np.ndarray):
            self._nps = data
            self._n_ch = 1 if data.ndim < 2 else data.shape[0]
    
    def from_dh(self, dh, group="noise", dataset="nps"):
        self._nps = dh.get(group, dataset)
        self._n_ch = 1 if self._nps.ndim < 2 else self._nps.shape[0]
        return self
        
    def to_dh(self, dh, group="noise", dataset="nps", **kwargs):
        dh.set(group, **{dataset: self._nps}, **kwargs)
        
    def from_file(self, file: str):
        ...
        
    def to_file(self):
        ...

    def show(self, dt: int = None, **kwargs):
        if 'xscale' not in kwargs.keys(): kwargs['xscale'] = 'log'
        if 'yscale' not in kwargs.keys(): kwargs['yscale'] = 'log'

        if dt is not None:
            if 'x' in kwargs.keys():
                print("Setting data for the x-axis overrides input 'dt'.")
            else:
                # TODO: convert to proper frequencies
                kwargs['x'] = None

        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(self._array):
                y[f'channel {i}'] = channel
        else:
            y = self._array

        return Line(y, **kwargs)
    
    @property
    def _array(self):
        return self._nps
    
    @_array.setter
    def _array(self, array):
        if self._nps.size == 0:
            self._nps = array
            self._n_ch = 1 if array.ndim < 2 else array.shape[0]
        else:
            raise Exception("NPS._array can only be set as long as it is empty.")
    
    @property
    def _n_channels(self):
        return self._n_ch
    
class OF(ArrayWithBenefits):
    def __init__(self, *args: Any):
        if isinstance(args, np.ndarray):
            self._of = args
            self._n_ch = 1 if args.ndim < 2 else args.shape[0]
        elif type(args) is tuple and len(args) == 2:
            bool_sev = [isinstance(k, SEV) for k in args]
            bool_nps = [isinstance(k, NPS) for k in args]

            if not any(bool_sev) or not any(bool_nps):
                raise TypeError(f"If 2 arguments are parsed, one of them has to be of class 'NPS' and one of class 'SEV', not {type(args[0])} and {type(args[1])}.")
            
            sev, nps = args[bool_sev.index(True)], args[bool_nps.index(True)]

            if sev._n_channels != nps._n_channels:
                raise Exception(f"SEV and NPS must have the same number of channels. Numbers received: ({sev._n_channels},{nps._n_channels})")
            
            self._n_ch = sev._n_channels

            # Cast to numpy array here (this will get rid of the SEV and NPS character)
            sev, nps = np.array(sev), np.array(nps)

            # Maximum time in samples
            t_m = np.argmax(sev, axis=-1)
            # Cast t_m into a column vector such that vectorization works
            if t_m.ndim > 0: t_m = t_m[:, None]

            # np.fft.rfftfreq is [0, 1/n, 2/n, ... 1/2-1/n, 1/2], i.e. in units of cycles/sample
            # But the definition of the inverse Fourier transform in GATTI and MANFREDI reveals 
            # that they are using an omega given in rad/s. 
            # To match this convention, we have to multiply by 2*pi to go from 
            # frequency to angular frequency
            omega = 2*np.pi*np.fft.rfftfreq(sev.shape[-1])

            H = np.fft.rfft(sev).conjugate()*np.exp(-1j*t_m*omega)/nps

            # Normalize to height of SEV
            maxima = np.max(OptimumFiltering(H)(sev), axis=-1)
            # Cast maxima into a column vector such that vectorization works
            if maxima.ndim > 0: maxima = maxima[:, None]
            
            self._of = H/maxima

        elif type(args) is tuple and len(args) == 0:
            self._of = np.empty(0)
            self._n_ch = 0
        else:
            raise TypeError(f"Unsupported input arguments {args}")
    
    def from_dh(self, dh, group="optimumfilter", dataset="optimumfilter"):
        self._of = dh.get(group, dataset+'_real') + 1j*dh.get(group, dataset+'_imag')
        self._n_ch = 1 if self._of.ndim < 2 else self._of.shape[0]
        return self
        
    def to_dh(self, dh, group="optimumfilter", dataset="optimumfilter", **kwargs):
        dh.set(group, **{dataset+"_real": np.real(self._of), 
                         dataset+"_imag": np.imag(self._of)}, 
                         **kwargs)
        
    def from_file(self, file: str):
        ...
        
    def to_file(self):
        ...

    def show(self, dt: int = None, **kwargs):
        if 'xscale' not in kwargs.keys(): kwargs['xscale'] = 'log'
        if 'yscale' not in kwargs.keys(): kwargs['yscale'] = 'log'

        if dt is not None:
            if 'x' in kwargs.keys():
                print("Setting data for the x-axis overrides input 'dt'.")
            else:
                # TODO: convert to proper frequencies
                kwargs['x'] = None

        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(self._array):
                y[f'abs(channel {i})'] = np.abs(channel)
        else:
            y = np.abs(self._array)

        return Line(y, **kwargs)

    @property
    def _array(self):
        return self._of
    
    @_array.setter
    def _array(self, array):
        if self._of.size == 0:
            self._of = array
            self._n_ch = 1 if array.ndim < 2 else array.shape[0]
        else:
            raise Exception("OF._array can only be set as long as it is empty.")
    
    @property
    def _n_channels(self):
        return self._n_ch