from abc import ABC, abstractmethod

import numpy as np

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
    
    def __getitem__(self, key):
        out = self.__class__()
        out._array = self._array.__getitem__(key)
        return out
    
    def __setitem__(self, key, val):
        self._array[key] = val
    
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