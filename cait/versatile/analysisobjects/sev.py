from typing import Union
import os

import numpy as np
from tqdm.auto import tqdm

from .arraywithbenefits import ArrayWithBenefits
from ..iterators.iteratorbase import IteratorBaseClass
from ..eventfunctions.processing.removebaseline import RemoveBaseline
from ..plot.basic.line import Line

from ...data import write_xy_file

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
            data = data.flatten().with_processing(RemoveBaseline())
            if len(data) > 1000:
                mean_pulse = np.zeros_like(data.grab(0))
                with data:
                    for ev in tqdm(data, delay=5):
                        mean_pulse+=ev
                mean_pulse/=len(data)
            else:
                mean_pulse = np.mean(data, axis=0)
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
    
    @classmethod
    def from_dh(cls, dh, group: str = "stdevent", dataset: str = "event"):
        """
        Construct SEV from DataHandler. 

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the SEV is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the SEV is stored.
        :type dataset: str

        :return: Instance of SEV.
        :rtype: SEV
        """
        return cls(dh.get(group, dataset))
        
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
        
    @classmethod
    def from_file(cls, fname: str, src_dir: str = ''):
        """
        Construct SEV from xy-file.

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

        arr = np.genfromtxt(fpath, skip_header=line_nr, delimiter="\t").T

        return cls(arr)
        
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

    def show(self, dt_us: int = None, **kwargs):
        """
        Plot SEV for all channels. To inspect just one channel, you can index SEV first and call `.show` on the slice.

        :param dt_us: Length of a sample in microseconds. If provided, the x-axis is a microsecond axis. Otherwise it's the sample index.
        :type dt_us: int
        :param kwargs: Keyword arguments passed on to `cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")
        
        if 'x' not in kwargs.keys():
            if dt_us is not None:
                kwargs['x'] = dt_us/1000*(np.arange(self.shape[-1]) - int(self.shape[-1]/4))
            else:
                kwargs['x'] = np.arange(self.shape[-1])
            
        if 'xlabel' not in kwargs.keys():
            if dt_us is not None: 
                kwargs['xlabel'] = "Time (ms)"
            else:
                kwargs['xlabel'] = "Data Index"

        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(self._array):
                y[f'channel {i}'] = channel
                # if hasattr(self, "_fit_pars"):
                #     y[f'channel {i} fit'] = self.fit_model[i](kwargs['x'])
        else:
            # if hasattr(self, "_fit_pars"):
            #     y = dict(event=self._array, fit=self.fit_model(kwargs['x']))
            # else:
            #     y = self._array
            y = self._array

        return Line(y, **kwargs)

 # Unfinished   
    # def fit(self, t: np.ndarray, n_comp: int = 2, **kwargs):
    #     if "p0" not in kwargs.keys():
    #         kwargs["p0"] = [0, 
    #                         *[1/10**k for k in range(n_comp)], 
    #                         *[100/10**(k+1) for k in range(n_comp+1)]
    #                         ]

    #     if self._n_channels > 1:
    #         self._fit_pars = []
    #         for channel in self._array:
    #             pars, _ = sp.optimize.curve_fit(pulse_template, t, channel, **kwargs)
    #             self._fit_pars.append(pars)
    #     else:
    #         self._fit_pars, _ = sp.optimize.curve_fit(pulse_template, t, self._array, **kwargs)

    # @property
    # def fit_pars(self):
    #     if not hasattr(self, "_fit_pars"):
    #         raise KeyError("No fit parameters are available. Make sure to fit the SEV first.")
        
    #     return self._fit_pars
    
    # @property
    # def fit_model(self):
    #     if self._n_channels > 1:
    #         models = []
    #         for fp in self.fit_pars:
    #             models.append(lambda t: pulse_template(t, *fp))
    #         return models
    #     else:
    #         return lambda t: pulse_template(t, *self.fit_pars)
        
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