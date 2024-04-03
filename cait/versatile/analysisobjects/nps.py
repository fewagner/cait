from typing import Union
import os

import numpy as np

from .arraywithbenefits import ArrayWithBenefits
from ..iterators.iteratorbase import IteratorBaseClass
from ..eventfunctions.processing.removebaseline import RemoveBaseline
from ..functions.apply import apply
from ..plot.basic.line import Line

from ...data import write_xy_file

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
    
    @classmethod
    def from_dh(cls, dh, group: str = "noise", dataset: str = "nps"):
        """
        Construct NPS from DataHandler. 

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the NPS is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NPS is stored.
        :type dataset: str

        :return: Instance of NPS.
        :rtype: NPS
        """
        return cls(dh.get(group, dataset))
        
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