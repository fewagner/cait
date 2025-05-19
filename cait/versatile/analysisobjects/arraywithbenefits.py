from abc import ABC, abstractmethod

import numpy as np

class ArrayWithBenefits(ABC, np.lib.mixins.NDArrayOperatorsMixin):
    def __repr__(self):
        return f"{self.__class__.__name__}({self._array})"
    
    def __len__(self):
        return len(self._array)
    
    def __array__(self, *args, **kwargs):
        # we do not handle additional arguments that might be provided
        # by numpy calls
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
            out._dt_us = self.dt_us
        else:
            out = f[method](*args, **kwargs)
        
        return out
    
    def __getitem__(self, key):
        out = self.__class__()
        out._array = self._array.__getitem__(key)
        out._dt_us = self.dt_us
        return out
    
    def __setitem__(self, key, val):
        self._array[key] = val

    # redirect all attribute calls to underlying numpy object
    # (if not explicitly defined by ArrayWithBenefits)
    def __getattr__(self, name):
        if hasattr(np.ndarray, name):
            return np.array(self._array).__getattribute__(name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'.")
    
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
        
    @property
    @abstractmethod
    def dt_us(self):
        ...