import os
import itertools
from typing import Union, Any
from abc import ABC, abstractmethod

import numpy as np

from ..iterators import IteratorBaseClass, apply
from ..functions import OptimumFiltering, RemoveBaseline
from ..plot import Line

from ...data import write_xy_file

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

class SEV(ArrayWithBenefits):
    """
    Object representing a Standard Event (SEV). It can either be created by averaging events from an `EventIterator`, from an `np.ndarray` or read from a DataHandler or xy-file.

    If created from an `EventIterator`, the (constant) baseline is removed automatically.

    :param data: The data to use for the SEV. If None, an empty SEV is created. If `np.ndarray`, each row in the array is interpreted as a SEV for separate channels. If iterator (possibly from multiple channels) a SEV is calculated by averaging the events returned by the iterator. Defaults to None.
    :type data: Union[np.array, Type[IteratorBaseClass]]
    """
    def __init__(self, data: Union[np.ndarray, IteratorBaseClass] = None):
        if data is None:
            self._sev = np.empty(0)
            self._n_ch = 0
        elif isinstance(data, IteratorBaseClass):
            mean_pulse = np.mean(apply(RemoveBaseline(), data), axis=0)
            # Normalize
            maxima = np.max(mean_pulse, axis=-1)
            # Cast maxima into a column vector such that vectorization works
            if maxima.ndim > 0: maxima = maxima[:, None]

            self._sev = mean_pulse/maxima
            if self._sev.ndim > 1:
                self._n_ch = self._sev.shape[0]
                if self._n_ch == 1: self._sev = self._sev.flatten()
            else:
                self._n_ch = 1
        elif isinstance(data, np.ndarray):
            self._sev = data
            if self._sev.ndim > 1:
                self._n_ch = self._sev.shape[0]
                if self._n_ch == 1: self._sev = self._sev.flatten()
            else:
                self._n_ch = 1
        else:
            raise ValueError(f"Unsupported datatype '{type(data)}' for input argument 'data'.")
    
    def from_dh(self, dh, group: str = "stdevent", dataset: str = "event"):
        """
        Read SEV from DataHandler. 

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the SEV is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the SEV is stored.
        :type dataset: str

        :return: Instance of SEV.
        :rtype: SEV
        """
        self._sev = dh.get(group, dataset)
        
        if self._sev.ndim > 1:
            self._n_ch = self._sev.shape[0]
            if self._n_ch == 1: self._sev = self._sev.flatten()
        else:
            self._n_ch = 1

        return self
        
    def to_dh(self, dh, group: str = "stdevent", dataset: str = "event", **kwargs):
        """
        Save SEV to DataHandler. 

        :param dh: The DataHandler instance to write to.
        :type dh: DataHandler
        :param group: The HDF5 group where the SEV should be stored.
        :type group: str
        :param dataset: The HDF5 dataset where the SEV should be stored.
        :type dataset: str
        :param kwargs: Keyword arguments for `DataHandler.set`.
        :type kwargs: Any
        """
        data = self._sev[None,:] if self._n_channels == 1 else self._sev
        dh.set(group, **{dataset: data}, **kwargs)
        
    def from_file(self, fname: str, src_dir: str = ''):
        """
        Read SEV from xy-file.

        :param fname: Filename to look for (without file-extension)
        :type fname: str
        :param out_dir: Directory to look in. Defaults to '' which means searching current directory. Optional
        :type out_dir: str

        :return: Instance of SEV.
        :rtype: SEV
        """
        fpath = os.path.join(src_dir, fname + ".txt")

        # We read the file to find out how many header lines it has
        # The number found is the number of axis + title.
        with open(fpath, "r") as f:
            line_nr = 0
            for l in f.readlines():
                # Try converting to float (will fail for strings)
                try: 
                    float(l.split("\n")[0].split("\t")[0])
                    # Once we found a line which can be converted to float, we stop
                    break
                except ValueError: 
                    line_nr += 1

        self._sev = np.genfromtxt(fpath, skip_header=line_nr, delimiter="\t").T
        
        if self._sev.ndim > 1:
            self._n_ch = self._sev.shape[0]
            if self._n_ch == 1: self._sev = self._sev.flatten()
        else:
            self._n_ch = 1

        return self
        
    def to_file(self, fname: str, out_dir: str = ''):
        """
        Write SEV to xy-file.

        :param fname: Filename to use (without file-extension)
        :type fname: str
        :param out_dir: Directory to write to. Defaults to '' which means writing to current directory. Optional
        :type out_dir: str
        """
        if np.array_equal(self._array, np.empty(0)):
            raise Exception("Empty SEV cannot be saved.")

        fpath = os.path.join(out_dir, fname + ".txt")

        if self._n_ch > 1:
            data = [self._array[k] for k in range(self._n_ch)]
        else:
            data = [self._array]

        write_xy_file(fpath, 
                      data=data, 
                      title="Standard Event", 
                      axis=[f"Channel {k}" for k in range(self._n_ch)])

    def show(self, dt: int = None, **kwargs):
        """
        Plot SEV for all channels. To inspect just one channel, you can index SEV first and call `.show` on the slice.

        :param dt: Length of a sample in microseconds. If provided, the x-axis is a microsecond axis. Otherwise it's the sample index.
        :type dt: int
        :param kwargs: Keyword arguments passed on to `cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if dt is not None:
            if 'x' not in kwargs.keys():
                n = self.shape[-1]
                kwargs['x'] = dt/1000*(np.arange(n) - int(n/4))
            
        if 'xlabel' not in kwargs.keys():
            if dt is not None: 
                kwargs['xlabel'] = "Time (ms)"
            else:
                kwargs['xlabel'] = "Data Index"

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
    """
    Object representing a Noise Power Spectrum (NPS). It can either be created by averaging the Fourier transformed events from an `EventIterator`, from an `np.ndarray` or read from a DataHandler or xy-file.

    If created from an `EventIterator`, the (constant) baseline is removed automatically.

    :param data: The data to use for the NPS. If None, an empty NPS is created. If `np.ndarray`, each row in the array is interpreted as an NPS for separate channels. If iterator (possibly from multiple channels) an NPS is calculated by averaging the Fourier transformed events returned by the iterator. Defaults to None.
    :type data: Union[np.array, Type[IteratorBaseClass]]
    """
    def __init__(self, data: Union[np.ndarray, IteratorBaseClass] = None):
        if data is None:
            self._nps = np.empty(0)
            self._n_ch = 0
        elif isinstance(data, IteratorBaseClass):
            data = data.with_processing(RemoveBaseline())
            self._nps = np.mean(apply(lambda x: np.abs(np.fft.rfft(x))**2, data), axis=0)
            if self._nps.ndim > 1:
                self._n_ch = self._nps.shape[0]
                if self._n_ch == 1: self._nps = self._nps.flatten()
            else:
                self._n_ch = 1
        elif isinstance(data, np.ndarray):
            self._nps = data
            if self._nps.ndim > 1:
                self._n_ch = self._nps.shape[0]
                if self._n_ch == 1: self._nps = self._nps.flatten()
            else:
                self._n_ch = 1
        else:
            raise ValueError(f"Unsupported datatype '{type(data)}' for input argument 'data'.")
    
    def from_dh(self, dh, group: str = "noise", dataset: str = "nps"):
        """
        Read NPS from DataHandler. 

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the NPS is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NPS is stored.
        :type dataset: str

        :return: Instance of NPS.
        :rtype: NPS
        """
        self._nps = dh.get(group, dataset)
        
        if self._nps.ndim > 1:
            self._n_ch = self._nps.shape[0]
            if self._n_ch == 1: self._nps = self._nps.flatten()
        else:
            self._n_ch = 1

        return self
        
    def to_dh(self, dh, group: str = "noise", dataset: str = "nps", **kwargs):
        """
        Save NPS to DataHandler. 

        :param dh: The DataHandler instance to write to.
        :type dh: DataHandler
        :param group: The HDF5 group where the NPS should be stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NPS should be stored.
        :type dataset: str
        :param kwargs: Keyword arguments for `DataHandler.set`.
        :type kwargs: Any
        """
        data = self._nps[None,:] if self._n_channels == 1 else self._nps
        dh.set(group, **{dataset: data}, **kwargs)
        
    def from_file(self, fname: str, src_dir: str = ''):
        """
        Read NPS from xy-file.

        :param fname: Filename to look for (without file-extension)
        :type fname: str
        :param out_dir: Directory to look in. Defaults to '' which means searching current directory. Optional
        :type out_dir: str

        :return: Instance of NPS.
        :rtype: NPS
        """
        fpath = os.path.join(src_dir, fname + ".txt")

        # We read the file to find out how many header lines it has
        # The number found is the number of axis + title.
        with open(fpath, "r") as f:
            line_nr = 0
            for l in f.readlines():
                # Try converting to float (will fail for strings)
                try: 
                    float(l.split("\n")[0].split("\t")[0])
                    # Once we found a line which can be converted to float, we stop
                    break
                except ValueError: 
                    line_nr += 1

        self._nps = np.genfromtxt(fpath, skip_header=line_nr, delimiter="\t").T
        
        if self._nps.ndim > 1:
            self._n_ch = self._nps.shape[0]
            if self._n_ch == 1: self._nps = self._nps.flatten()
        else:
            self._n_ch = 1

        return self
        
    def to_file(self, fname: str, out_dir: str = ''):
        """
        Write NPS to xy-file.

        :param fname: Filename to use (without file-extension)
        :type fname: str
        :param out_dir: Directory to write to. Defaults to '' which means writing to current directory. Optional
        :type out_dir: str
        """
        if np.array_equal(self._array, np.empty(0)):
            raise Exception("Empty NPS cannot be saved.")

        fpath = os.path.join(out_dir, fname + ".txt")

        if self._n_ch > 1:
            data = [self._array[k] for k in range(self._n_ch)]
        else:
            data = [self._array]

        write_xy_file(fpath, 
                      data=data, 
                      title="Noise Power Spectrum", 
                      axis=[f"Channel {k}" for k in range(self._n_ch)])

    def show(self, dt: int = None, **kwargs):
        """
        Plot NPS for all channels. To inspect just one channel, you can index NPS first and call `.show` on the slice.

        :param dt: Length of a sample in microseconds. If provided, the x-axis is a frequency axis. Otherwise it's the sample index.
        :type dt: int
        :param kwargs: Keyword arguments passed on to `cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if 'xscale' not in kwargs.keys(): kwargs['xscale'] = 'log'
        if 'yscale' not in kwargs.keys(): kwargs['yscale'] = 'log'

        if dt is not None:
            if 'x' not in kwargs.keys():
                n = 2*(self.shape[-1]-1)
                kwargs['x'] = np.fft.rfftfreq(n, dt/1e6)

        if 'xlabel' not in kwargs.keys():
            if dt is not None: 
                kwargs['xlabel'] = "Frequency (Hz)"
            else:
                kwargs['xlabel'] = "Data Index"   
        
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
    """
    Object representing an Optimum Filter (OF). It can either be created from a Standard Event (SEV) and a Noise Power Spectrum (NPS), from an `np.ndarray` or read from a DataHandler or xy-file.

    :param args: The data to use for the OF. If None, an empty OF is created. If `np.ndarray`, each row in the array is interpreted as an OF for separate channels. If instances of class:`SEV` and class:`NPS`, the OF is calculated from them. Defaults to None.
    :type data: Any

    >>> sev = vai.SEV().from_dh(dh)
    >>> nps = vai.NPS().from_dh(dh)
    >>> of = vai.OF(sev, nps)
    """
    def __init__(self, *args: Any):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            self._of = args[0]
            if self._of.ndim > 1:
                self._n_ch = self._of.shape[0]
                if self._n_ch == 1: self._of = self._of.flatten()
            else:
                self._n_ch = 1

        elif len(args) == 2:
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

            # The cait function calc_nps has an argument `force_zero` which sets
            # the first entry of the NPS to 0. If the user uses such an NPS to 
            # create an OF, we set its first entry to zero to stay in line with
            # the standard cait routines.
            if self._n_ch == 1:
                if nps[0] == 0: 
                    s1, s2 = slice(1, None), slice(1, None)
                else: 
                    s1, s2 = slice(None, None), slice(None, None)
            else:
                if np.any(nps[:,0] == 0): 
                    s1, s2 = (slice(None, None), slice(1, None)), slice(1, None)
                else: 
                    s1, s2 = (slice(None, None), slice(None, None)), slice(None, None)
            
            H = np.zeros(nps.shape, dtype=complex)
            H[s1] = np.fft.rfft(sev[s1]).conjugate()*np.exp(-1j*t_m*omega[s2])/nps[s1]

            # Normalize to height of SEV
            maxima = np.max(OptimumFiltering(H)(sev), axis=-1)
            # Cast maxima into a column vector such that vectorization works
            if maxima.ndim > 0: maxima = maxima[:, None]
            
            self._of = H/maxima

        elif len(args) == 0:
            self._of = np.empty(0)
            self._n_ch = 0
        else:
            raise TypeError(f"Unsupported input arguments {args}")
    
    def from_dh(self, dh, group: str = "optimumfilter", dataset: str = "optimumfilter*"):
        """
        Read OF from DataHandler. 

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the OF is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the OF is stored. The star `*` denotes the position of the suffixes '_real' and '_imag'.
        :type dataset: str

        :return: Instance of OF.
        :rtype: OF
        """
        if "*" not in dataset: dataset += "*"
        ds_prefix, ds_suffix = dataset.split("*")

        self._of = dh.get(group, ds_prefix+'_real'+ds_suffix) + 1j*dh.get(group, ds_prefix+'_imag'+ds_suffix)
        
        if self._of.ndim > 1:
            self._n_ch = self._of.shape[0]
            if self._n_ch == 1: self._of = self._of.flatten()
        else:
            self._n_ch = 1
            
        return self
        
    def to_dh(self, dh, group: str = "optimumfilter", dataset: str = "optimumfilter*", **kwargs):
        """
        Save OF to DataHandler. 

        :param dh: The DataHandler instance to write to.
        :type dh: DataHandler
        :param group: The HDF5 group where the OF should be stored.
        :type group: str
        :param dataset: The HDF5 dataset where the OF should be stored. The star `*` denotes the position of the suffixes '_real' and '_imag'.
        :type dataset: str
        :param kwargs: Keyword arguments for `DataHandler.set`.
        :type kwargs: Any
        """
        if "*" not in dataset: dataset += "*"
        ds_prefix, ds_suffix = dataset.split("*")

        data = self._of[None,:] if self._n_channels == 1 else self._of

        dh.set(group, **{ds_prefix+"_real"+ds_suffix: np.real(data), 
                         ds_prefix+"_imag"+ds_suffix: np.imag(data)}, 
                         **kwargs)
        
    def from_file(self, fname: str, src_dir: str = ''):
        """
        Read OF from xy-file.

        :param fname: Filename to look for (without file-extension)
        :type fname: str
        :param out_dir: Directory to look in. Defaults to '' which means searching current directory. Optional
        :type out_dir: str

        :return: Instance of OF.
        :rtype: OF
        """
        fpath = os.path.join(src_dir, fname + ".txt")

        # We read the file to find out how many header lines it has
        # The number found is the number of axis + title.
        with open(fpath, "r") as f:
            line_nr = 0
            for l in f.readlines():
                # Try converting to float (will fail for strings)
                try: 
                    float(l.split("\n")[0].split("\t")[0])
                    # Once we found a line which can be converted to float, we stop
                    break
                except ValueError: 
                    line_nr += 1

        data = np.genfromtxt(fpath, skip_header=line_nr, delimiter="\t").T

        if not (data.ndim%2)==0 or data.ndim==0:
            raise Exception("Compatible files must have an even number of data columns (containing real and imaginary part of the OF, respectively)")

        self._of = data[::2] + 1j*data[1::2]
        
        if self._of.ndim > 1:
            self._n_ch = self._of.shape[0]
            if self._n_ch == 1: self._of = self._of.flatten()
        else:
            self._n_ch = 1

        return self
        
    def to_file(self, fname: str, out_dir: str = ''):
        """
        Write OF to xy-file.

        :param fname: Filename to use (without file-extension)
        :type fname: str
        :param out_dir: Directory to write to. Defaults to '' which means writing to current directory. Optional
        :type out_dir: str
        """
        if np.array_equal(self._array, np.empty(0)):
            raise Exception("Empty OF cannot be saved.")

        fpath = os.path.join(out_dir, fname + ".txt")

        if self._n_ch > 1:
            data = [[self._array[k].real, self._array[k].imag] for k in range(self._n_ch)]
            data = list(itertools.chain.from_iterable(data))
            names = [[f"Real Channel {k}", f"Imag Channel {k}"] for k in range(self._n_ch)]
            names = list(itertools.chain.from_iterable(names))
        else:
            data = [self._array.real, self._array.imag]
            names = ["Real", "Imag"]

        write_xy_file(fpath, data=data, title="Optimum Filter", axis=names)

    def show(self, dt: int = None, **kwargs):
        """
        Plot OF for all channels. To inspect just one channel, you can index OF first and call `.show` on the slice.

        :param dt: Length of a sample in microseconds. If provided, the x-axis is a frequency axis. Otherwise it's the sample index.
        :type dt: int
        :param kwargs: Keyword arguments passed on to `cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if 'xscale' not in kwargs.keys(): kwargs['xscale'] = 'log'
        if 'yscale' not in kwargs.keys(): kwargs['yscale'] = 'log'

        if dt is not None:
            if 'x' not in kwargs.keys():
                n = 2*(self.shape[-1]-1)
                kwargs['x'] = np.fft.rfftfreq(n, dt/1e6)

        if 'xlabel' not in kwargs.keys():
            if dt is not None: 
                kwargs['xlabel'] = "Frequency (Hz)"
            else:
                kwargs['xlabel'] = "Data Index"

        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(self._array):
                y[f'|channel {i}|'] = np.abs(channel)
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