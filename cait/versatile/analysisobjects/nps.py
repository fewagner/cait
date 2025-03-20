from typing import Union
import os

import numpy as np
from tqdm.auto import tqdm

from .arraywithbenefits import ArrayWithBenefits
from .helper import is_array_like
from ..iterators.iteratorbase import IteratorBaseClass
from ..eventfunctions.processing.removebaseline import RemoveBaseline
from ..plot.basic.line import Line

from ...data import write_xy_file

class NPS(ArrayWithBenefits):
    """
    Object representing a Noise Power Spectrum (NPS) and Cross Power Spectrum (CPS). It can either be created by averaging the Fourier transformed events from an `EventIterator`, from an `np.ndarray` or read from a DataHandler or xy-file.

    If created from an `EventIterator`, the (constant) baseline is removed automatically.
    To improve the quality of the NPS, a window function is often applied to the noise traces before performing the Fourier transform and averaging (see Numerical Recipes by Press, Teukolsky, Vetterling, Flannery chapter 13.4.1). This can only be achieved when we still have the original noise traces, i.e. when we construct the NPS from an iterator. Instead of just a bare iterator ``it`` you can pass the iterator ``it.with_processing([vai.RemoveBaseline(), vai.TukeyFiltering()])`` to ``NPS``. 

    :param data: The data to use for the NPS. If None, an empty NPS is created. If `np.ndarray`, each row in the array is interpreted as an NPS for separate channels. If iterator (possibly from multiple channels) an NPS is calculated by averaging the Fourier transformed events returned by the iterator. Defaults to None.
    :type data: Union[np.array, Type[IteratorBaseClass]]
    """
    def __init__(self, data: Union[np.ndarray, IteratorBaseClass] = None):
        if data is None:
            self._nps = np.empty(0)
            self._cps = np.empty(0)
            self._n_ch = 0
        elif isinstance(data, IteratorBaseClass):
            data = data.flatten().with_processing([RemoveBaseline(), 
                                                lambda x: np.abs(np.fft.rfft(x))])
            cor_id = [(0, 1), (1, 2), (2, 0)]
            if len(data) > 1000:
                self._nps = np.zeros_like(data.grab(0))
                self._cps = np.zeros_like(data.grab(0))
                self._cor = np.zeros_like(data.grab(0))
                with data:
                    for ev in tqdm(data, delay=5):
                        self._nps+=ev**2
                        if self._nps.ndim > 1:
                            for i in range(self._nps.shape[0]):
                                self._cps[i]+=ev[cor_id[i][0]]*ev[cor_id[i][1]]
                self._nps/=len(data)
                self._cps/=len(data)
                if self._nps.ndim > 1:
                    for i in range(self._nps.shape[0]):
                        self._cor[i] = self._cps[i]**2/(self._nps[cor_id[i][0]]*self._nps[cor_id[i][1]])
            else:
                with data:
                    self._nps = np.mean(data**2, axis=0)
                    for ev in tqdm(data, delay=5):
                        if self._nps.ndim > 1:
                            for i in range(self.n_ch):
                                self._cps[i]+=ev[cor_id[i][0]]*ev[cor_id[i][1]]
                self._cps/=len(data)
                if self._nps.ndim > 1:
                    for i in range(self._nps.shape[0]):
                        self._cor[i] = self._cps[i]**2/(self._nps[cor_id[i][0]]*self._nps[cor_id[i][1]])
            if self._nps.ndim > 1:
                self._n_ch = self._nps.shape[0]
                if self._n_ch == 1: self._nps = self._nps.flatten()
            else:
                self._n_ch = 1
        elif isinstance(data, np.ndarray) or is_array_like(data):
            self._nps = np.array(data)
            if self._nps.ndim > 1:
                self._n_ch = self._nps.shape[0]
                if self._n_ch == 1: self._nps = self._nps.flatten()
            else:
                self._n_ch = 1
        else:
            raise ValueError(f"Unsupported datatype '{type(data)}' for input argument 'data'.")
    
    @classmethod
    def from_dh(cls, dh, group: str = "noise", dataset: str = 'nps'):   
        """
        Construct NPS from DataHandler. 

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the NPS is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NPS/CPS is stored.
        :type dataset: str

        :return: Instance of NPS.
        :rtype: NPS
        """
        if dataset == 'both':
            sets = ['nps', 'cps']
            bools = [dh.exists('noise', sets[i]) for i in range(2)]
            if bools[0]*bools[1]:
                return cls(dh.get(group, sets[0])), cls(dh.get(group, sets[1]))
            else:
                raise KeyError(f"Unable to synchronously open object (object '{sets[bools.index(False)]}' doesn't exist)")
        else:
            return cls(dh.get(group, dataset))
        
    def to_dh(self, dh, group: str = "noise", dataset: str = 'nps', **kwargs):
        """
        Save NPS and CSD to DataHandler. 

        :param dh: The DataHandler instance to write to.
        :type dh: DataHandler
        :param group: The HDF5 group where the NPS should be stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NPS should be stored.
        :type dataset: str
        :param kwargs: Keyword arguments for `DataHandler.set`.
        :type kwargs: Any
        """
        data_nps = self._nps[None,:] if self._n_channels == 1 else self._nps
        data_cps = self._cps[None,:] if self._n_channels == 1 else self._cps
        if dataset == 'both':
            dh.set(group, **{'nps': data_nps}, **kwargs)
            dh.set(group, **{'cps': data_cps}, **kwargs)
        else:
            dh.set(group, **{dataset: data_nps}, **kwargs)
        
    @classmethod
    def from_file(cls, fname: str, src_dir: str = ''):
        """
        Construct NPS from xy-file.

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

        arr = np.genfromtxt(fpath, skip_header=line_nr, delimiter="\t").T

        return cls(arr)
        
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

    def show(self, dt_us: int = None, **kwargs):
        """
        Plot NPS for all channels. To inspect just one channel, you can index NPS first and call `.show` on the slice.

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

        if 'ylabel' not in kwargs.keys():
            if dt_us is not None: 
                kwargs['ylabel'] = "Noise Power Density (V²/Hz)"
            else:
                kwargs['ylabel'] = "Noise Power Density (a.u.)"

        if dt_us is not None:
            _array = self._array/int(1e6/dt_us)/self._array.shape[-1]
        else:
            _array = self._array
        
        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(_array):
                y[f'channel {i}'] = channel
        else:
            y = _array

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