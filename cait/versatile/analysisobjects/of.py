from typing import Any
import itertools
import os

import numpy as np

from .arraywithbenefits import ArrayWithBenefits
from .sev import SEV
from .nps import NPS
from ..plot.basic.line import Line
from ..eventfunctions.processing.optimumfiltering import OptimumFiltering

from ...data import write_xy_file

class OF(ArrayWithBenefits):
    """
    Object representing an Optimum Filter (OF). It can either be created from a Standard Event (SEV) and a Noise Power Spectrum (NPS), from an `np.ndarray` or read from a DataHandler or xy-file.

    :param args: The data to use for the OF. If None, an empty OF is created. If `np.ndarray`, each row in the array is interpreted as an OF for separate channels. If instances of class:`SEV` and class:`NPS`, the OF is calculated from them. Defaults to None.
    :type data: Any

    .. code-block:: python
    
        sev = vai.SEV().from_dh(dh)
        nps = vai.NPS().from_dh(dh)
        of = vai.OF(sev, nps)
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

            # In any case, we force the 0 component of the filter kernel to 0
            # (this shifts the signal to 0)
            H[...,0] = 0

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
    
    @classmethod
    def from_dh(cls, dh, group: str = "optimumfilter", dataset: str = "optimumfilter*"):
        """
        Construct OF from DataHandler. 

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

        arr = dh.get(group, ds_prefix+'_real'+ds_suffix) + 1j*dh.get(group, ds_prefix+'_imag'+ds_suffix)
            
        return cls(arr)
        
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
        
    @classmethod
    def from_file(cls, fname: str, src_dir: str = ''):
        """
        Construct OF from xy-file.

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

        arr = data[::2] + 1j*data[1::2]
        
        return cls(arr)
        
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

    def show(self, dt_us: int = None, **kwargs):
        """
        Plot OF for all channels. To inspect just one channel, you can index OF first and call `.show` on the slice.

        :param dt_us: Length of a sample in microseconds. If provided, the x-axis is a frequency axis. Otherwise it's the sample index.
        :type dt_us: int
        :param kwargs: Keyword arguments passed on to `cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if 'xscale' not in kwargs.keys(): kwargs['xscale'] = 'log'
        if 'yscale' not in kwargs.keys(): kwargs['yscale'] = 'log'

        if dt_us is not None:
            if 'x' not in kwargs.keys():
                n = 2*(self.shape[-1]-1)
                kwargs['x'] = np.fft.rfftfreq(n, dt_us/1e6)

        if 'xlabel' not in kwargs.keys():
            if dt_us is not None: 
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