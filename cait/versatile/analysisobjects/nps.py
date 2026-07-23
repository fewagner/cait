import json
import os
import re
from typing import Union

import numpy as np
from tqdm.auto import tqdm

import cait as ai

from ..eventfunctions.processing.removebaseline import RemoveBaseline
from ..eventfunctions.processing.tukey import TukeyWindow
from ..iterators.iteratorbase import IteratorBaseClass
from ..plot.basic.line import Line
from .arraywithbenefits import ArrayWithBenefits
from .helper import is_array_like


class NPS(ArrayWithBenefits):
    """
    Object representing a Noise Power Spectrum (NPS). It can either be created by averaging the Fourier transformed events from an `EventIterator`, from an `np.ndarray` or read from a DataHandler or xy-file. If created from an `EventIterator`, a proper normalisation to physical units (V²/Hz) is performed automatically.

    If created from an `EventIterator`, the (constant) baseline is removed automatically.
    To improve the quality of the NPS, a window function is often applied to the noise traces before performing the Fourier transform and averaging (see Numerical Recipes by Press, Teukolsky, Vetterling, Flannery chapter 13.4.1). This can only be achieved when we still have the original noise traces, i.e. when we construct the NPS from an iterator. Instead of just a bare iterator ``it`` you can pass the iterator ``it.with_processing([vai.RemoveBaseline(), vai.TukeyWindow()])`` to ``NPS``.

    :param data: The data to use for the NPS. If None, an empty NPS is created. If `np.ndarray`, each row in the array is interpreted as an NPS for separate channels. If iterator (possibly from multiple channels) an NPS is calculated by averaging the Fourier transformed events returned by the iterator. Defaults to None.
    :type data: Union[np.array, Type[IteratorBaseClass]]
    :param dt_us: The microsecond timebase used in the recording. Only necessary if the input is an array (because it can be automatically inferred if constructed from an EventIterator, a file, or a DataHandler). Defaults to None (i.e. automatic detection if possible).
    :type dt_us: int
    """

    def __init__(
        self, data: Union[np.ndarray, IteratorBaseClass] = None, dt_us: int = None
    ):
        if data is None:
            self._nps = np.empty(0)
            self._n_ch = 0
            self._dt_us = dt_us
        elif isinstance(data, IteratorBaseClass):
            self._dt_us = data.dt_us
            data = data.flatten().with_processing(
                [RemoveBaseline(), lambda x: np.abs(np.fft.rfft(x)) ** 2]
            )
            if len(data) > 1000:
                self._nps = np.zeros_like(data.grab(0))
                with data:
                    for ev in tqdm(data, delay=5):
                        self._nps += ev
                self._nps /= len(data)
            else:
                with data:
                    self._nps = np.mean(data, axis=0)
            if self._nps.ndim > 1:
                self._n_ch = self._nps.shape[0]
                if self._n_ch == 1:
                    self._nps = self._nps.flatten()
            else:
                self._n_ch = 1

            # Normalise (according to Teukolsky, + time normalisation to "per Hz", i.e. divide by record time)
            N = 2 * (self._nps.shape[-1] - 1)
            w = np.ones(N)
            for f in data[:].pop_processing():
                if isinstance(f, TukeyWindow):
                    w = f(w)
            W = N * np.sum(w**2)

            # 1/(record time (in seconds)) = frequency bin size in Hz
            delta_f = 1 / (N * data.dt_us * 1e-6)

            # factor 2 for rFFT
            self._nps = 2 * self._nps / W / delta_f

        elif isinstance(data, np.ndarray) or is_array_like(data):
            if dt_us is None:
                raise ValueError(
                    "If NPS is constructed from array(-like) input, the microsecond timebase of the recording (dt_us) has to be specified."
                )
            self._dt_us = dt_us
            self._nps = np.array(data)
            if self._nps.ndim > 1:
                self._n_ch = self._nps.shape[0]
                if self._n_ch == 1:
                    self._nps = self._nps.flatten()
            else:
                self._n_ch = 1
        else:
            raise ValueError(
                f"Unsupported datatype '{type(data)}' for input argument 'data'."
            )

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
        return cls(dh.get(group, dataset), dt_us=dh.dt_us)

    def to_dh(self, dh, group: str = "noise", dataset: str = "nps", **kwargs):
        """
        Save NPS to DataHandler.

        :param dh: The DataHandler instance to write to.
        :type dh: DataHandler
        :param group: The HDF5 group where the NPS should be stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NPS should be stored.
        :type dataset: str
        :param kwargs: Keyword arguments for :func:`cait.DataHandler.set`.
        :type kwargs: Any
        """
        if self._dt_us != dh.dt_us:
            raise ValueError(
                f"Timebase of NPS ({self._dt_us}) does not match the one of DataHandler ({dh.dt_us})."
            )
        data = self._nps[None, :] if self._n_channels == 1 else self._nps
        dh.set(group, **{dataset: data}, **kwargs)

    @classmethod
    def from_file(cls, fname: str, src_dir: str = ""):
        """
        Construct NPS from xy-file.

        :param fname: Filename to look for (without file-extension)
        :type fname: str
        :param src_dir: Directory to look in. Defaults to '' which means searching current directory. Optional
        :type src_dir: str

        :return: Instance of NPS.
        :rtype: NPS
        """
        fpath = os.path.join(src_dir, fname + ".xy")

        # We read the file to find out how many header lines it has
        with open(fpath, "r") as f:
            first_line = f.readline()

        header = re.findall(r"\{.*\}", first_line)
        if header and all([x in json.loads(header[0]) for x in ["dt_us", "n_ch"]]):
            info = json.loads(header[0])
            arr = np.loadtxt(fpath, skiprows=info["n_ch"] + 3)[:, 1:].T
            dt_us = info["dt_us"]
        else:
            content = np.loadtxt(fpath)
            arr = content[:, 1:].T
            dt_us = int(
                np.round(
                    1 / np.diff(content[:, 0])[0] / (2 * (arr.shape[-1] - 1)) * 1e6
                )
            )

        return cls(arr, dt_us=dt_us)

    def to_file(self, fname: str, out_dir: str = "", info_str: str = ""):
        """
        Write NPS to xy-file.

        :param fname: Filename to use (without file-extension)
        :type fname: str
        :param out_dir: Directory to write to. Defaults to '' which means writing to current directory. Optional
        :type out_dir: str
        :param info_str: An info string to be saved as a description of the NPS in the header of the file. Optional
        :type info_str: str
        """
        if np.array_equal(self._array, np.empty(0)):
            raise Exception("Empty NPS cannot be saved.")

        fpath = os.path.join(out_dir, fname + ".xy")

        if self._n_ch > 1:
            data = [self.freq] + [self._array[k] for k in range(self._n_ch)]
        else:
            data = [self.freq, self._array]

        header = (
            "NPS "
            + json.dumps(
                {
                    "cait": ai.__version__,
                    "dt_us": self.dt_us,
                    "record_length": 2 * (self._nps.shape[-1] - 1),
                    "n_ch": self._n_channels,
                    **({"info": str(info_str)} if info_str else {})
                }
            )
            + "\n"
        )
        header += "\n".join(
            ["Frequency (Hz)"]
            + [f"Power Density {k} (V²/Hz)" for k in range(self._n_ch)]
            + [f"{len(self.freq)}"]
        )

        np.savetxt(fpath, np.array(data).T, header=header)

    def show(self, **kwargs):
        """
        Plot NPS for all channels. To inspect just one channel, you can index NPS first and call `.show` on the slice.

        :param kwargs: Keyword arguments passed on to :class:`cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")

        if "xscale" not in kwargs.keys():
            kwargs["xscale"] = "log"
        if "yscale" not in kwargs.keys():
            kwargs["yscale"] = "log"

        if "x" not in kwargs.keys():
            kwargs["x"] = self.freq

        if "xlabel" not in kwargs.keys():
            kwargs["xlabel"] = "Frequency (Hz)"

        if "ylabel" not in kwargs.keys():
            kwargs["ylabel"] = "Noise Power Density (V²/Hz)"

        if self._n_channels > 1:
            y = dict()
            for i, channel in enumerate(self._nps):
                y[f"channel {i}"] = channel
        else:
            y = self._nps

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

    @property
    def dt_us(self):
        return self._dt_us

    @property
    def freq(self):
        if self.dt_us is not None:
            n = 2 * (self.shape[-1] - 1)
            return np.fft.rfftfreq(n, self.dt_us / 1e6)


class NCM(ArrayWithBenefits):
    """
    Object representing a Noise Covariance Matrix (NCM). The diagonal elements are Noise Power Spectra (NPS) of all channels, while the off-diagonal elements represent the Cross Power Spectra (CPS). The NCM is either constructed from an `EventIterator`, from an `np.ndarray` or read from a DataHandler or xy-file. If created from an `EventIterator`, a proper normalisation to physical units (V²/Hz) is performed automatically.

    If created from an `EventIterator`, the (constant) baseline is removed automatically.
    To improve the quality of the NCM, a window function is often applied to the noise traces before performing the Fourier transform and averaging (see Numerical Recipes by Press, Teukolsky, Vetterling, Flannery chapter 13.4.1). This can only be achieved when we still have the original noise traces, i.e. when we construct the NCM from an iterator. Instead of just a bare iterator ``it`` you can pass the iterator ``it.with_processing([vai.RemoveBaseline(), vai.TukeyWindow()])`` to ``NCM``.

    :param data: The data to use for the NCM. If None, an empty NCM is created. If `np.ndarray` of shape (n_channels, n_channels, n_freq), each diagonal entry (i,i,:) is interpreted as the NPS of channel i, and the off-diagonal elements (i,j,:) correspond to the CPS between channels i and j. If iterator an NCM is calculated by averaging the Fourier transformed events returned by the iterator. Defaults to None.
    :type data: Union[np.array, Type[IteratorBaseClass]]
    :param dt_us: The microsecond timebase used in the recording. Only necessary if the input is an array (because it can be automatically inferred if constructed from an EventIterator, a file, or a DataHandler). Defaults to None (i.e. automatic detection if possible).
    :type dt_us: int
    """

    def __init__(
        self, data: Union[np.ndarray, IteratorBaseClass] = None, dt_us: int = None
    ):
        if data is None:
            self._n_ch = 0
            self._dt_us = dt_us
            self._ncm = np.empty((0, 0, 0))
        elif isinstance(data, IteratorBaseClass):
            n_ch = data.n_channels
            self._n_ch = n_ch
            self._dt_us = data.dt_us
            nps = np.atleast_2d(NPS(data))
            self._ncm = np.zeros((n_ch, n_ch, nps.shape[-1]), dtype=np.complex128)

            data = data.flatten().with_processing(
                [RemoveBaseline(), lambda x: np.fft.rfft(x)]
            )

            with data:
                for ev in tqdm(data, delay=5):
                    # Calculate half of the off-diagonal elements
                    for i in range(n_ch):
                        for j in range(i + 1, n_ch):
                            self._ncm[i, j, :] += ev[i] * np.conjugate(ev[j])

            # Copy off-diagonal elements to the other half (just conjugate entries)
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    self._ncm[j, i, :] = np.conjugate(self._ncm[i, j])

            self._ncm /= len(data)

            # Normalise (according to Teukolsky, + time normalisation to "per Hz", i.e. divide by record time)
            N = 2 * (nps.shape[-1] - 1)
            w = np.ones(N)
            for f in data[:].pop_processing():
                if isinstance(f, TukeyWindow):
                    w = f(w)
            W = N * np.sum(w**2)

            # 1/(record time (in seconds)) = frequency bin size in Hz
            delta_f = 1 / (N * data.dt_us * 1e-6)

            # factor 2 for rFFT
            self._ncm = 2 * self._ncm / W / delta_f

            # Copy NPS to diagonal (has to be done AFTER the above because it is
            # already correctly normalized)
            for i in range(n_ch):
                self._ncm[i, i, :] = nps[i]

        elif isinstance(data, np.ndarray) or is_array_like(data):
            if dt_us is None:
                raise ValueError(
                    "If NCM is constructed from array(-like) input, the microsecond timebase of the recording (dt_us) has to be specified."
                )
            if np.ndim(data) != 3:
                raise ValueError(
                    "When the NCM is constructed form an array, the array has to be 3-dimensional."
                )
            self._dt_us = dt_us
            self._ncm = np.array(data)
            self._n_ch = self._ncm.shape[0]
        else:
            raise ValueError(
                f"Unsupported datatype '{type(data)}' for input argument 'data'."
            )

    @classmethod
    def from_dh(cls, dh, group: str = "noise", dataset: str = "ncm"):
        """
        Load NCM from DataHandler.

        :param dh: The DataHandler instance to read from.
        :type dh: DataHandler
        :param group: The HDF5 group where the NCM is stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NCM is stored.
        :type dataset: str

        :return: Instance of NCM.
        :rtype: NCM
        """
        return cls(dh.get(group, dataset), dt_us=dh.dt_us)

    def to_dh(self, dh, group: str = "noise", dataset: str = "ncm", **kwargs):
        """
        Save NCM to DataHandler.

        :param dh: The DataHandler instance to write to.
        :type dh: DataHandler
        :param group: The HDF5 group where the NCM should be stored.
        :type group: str
        :param dataset: The HDF5 dataset where the NCM should be stored.
        :type dataset: str
        :param kwargs: Keyword arguments for :func:`cait.DataHandler.set`.
        :type kwargs: Any
        """
        if self._dt_us != dh.dt_us:
            raise ValueError(
                f"Timebase of NCM ({self._dt_us}) does not match the one of DataHandler ({dh.dt_us})."
            )
        dh.set(group, **{dataset: self._ncm}, **kwargs, dtype=np.complex128)

    @classmethod
    def from_file(cls, fname: str, src_dir: str = ""):
        """
        Load NCM from xy-file.

        :param fname: Filename to look for (without file-extension)
        :type fname: str
        :param src_dir: Directory to look in. Defaults to '' which means searching current directory. Optional
        :type src_dir: str

        :return: Instance of NCM.
        :rtype: NCM
        """
        fpath = os.path.join(src_dir, fname + ".xy")

        # We read the file to find out how many header lines it has
        with open(fpath, "r") as f:
            first_line = f.readline()

        header = re.findall(r"\{.*\}", first_line)
        if header and all([x in json.loads(header[0]) for x in ["dt_us", "n_ch"]]):
            info = json.loads(header[0])
            n_ch = info["n_ch"]
            arr = np.loadtxt(fpath, skiprows=n_ch**2 + 1)[:, 1:].T
            dt_us = info["dt_us"]
        else:
            content = np.loadtxt(fpath)
            arr = content[:, 1:].T
            dt_us = int(
                np.round(
                    1 / np.diff(content[:, 0])[0] / (2 * (arr.shape[-1] - 1)) * 1e6
                )
            )
            n_ch = np.sqrt(arr.shape[0])
            if (n_ch % 1) != 0:
                raise Exception(
                    f"When loading NCM data from an xy-file without header, the number of columns must be n_channels**2 . The number of channels could not be inferred from teh number of columns ({arr.shape[0]})"
                )

        out = np.zeros((n_ch, n_ch, arr.shape[-1]), dtype=np.complex128)
        # Fill NPS (diagonal elements)
        for i in range(n_ch):
            out[i, i, :] = arr[i]

        # Reconstruct CPS from real and imaginary part and fill off-diagonals
        offdiag = arr[n_ch::2] + 1j * arr[n_ch + 1 :: 2]
        for where, elem in zip(
            [[i, j] for i in range(n_ch) for j in range(i + 1, n_ch)], offdiag
        ):
            out[tuple(where + [slice(None)])] = elem
            out[tuple(list(reversed(where)) + [slice(None)])] = np.conjugate(elem)

        return cls(out, dt_us=dt_us)

    def to_file(self, fname: str, out_dir: str = "", info_str: str = ""):
        """
        Write NCM to xy-file.

        :param fname: Filename to use (without file-extension)
        :type fname: str
        :param out_dir: Directory to write to. Defaults to '' which means writing to current directory. Optional
        :type out_dir: str
        :param info_str: An info string to be saved as a description of the NCM in the header of the file. Optional
        :type info_str: str
        """
        if np.array_equal(self._array, np.empty((0, 0, 0))):
            raise Exception("Empty NCM cannot be saved.")

        fpath = os.path.join(out_dir, fname + ".xy")

        data = (
            [self.freq]
            + [self._array[i, i].real for i in range(self._n_ch)]
            + [
                getattr(self._array[i, j], which)
                for i in range(self._n_ch)
                for j in range(i + 1, self._n_ch)
                for which in ["real", "imag"]
            ]
        )

        header = (
            "NCM "
            + json.dumps(
                {
                    "cait": ai.__version__,
                    "dt_us": self.dt_us,
                    "record_length": 2 * (self._ncm.shape[-1] - 1),
                    "n_ch": self._n_channels,
                    **({"info": str(info_str)} if info_str else {})
                }
            )
            + "\n"
        )
        header += "\n".join(
            ["Frequency (Hz)"]
            + [f"NPS Ch-{k} (V²/Hz)" for k in range(self._n_ch)]
            + [
                f"CPS Ch-({i},{j}) (V²/Hz), {which}"
                for i in range(self._n_ch)
                for j in range(i + 1, self._n_ch)
                for which in ["real", "imag"]
            ]
            + [f"{len(self.freq)}"]
        )

        np.savetxt(fpath, np.array(data).T, header=header)

    def show(self, **kwargs):
        """
        Plot NCM for all channels.

        :param kwargs: Keyword arguments passed on to :class:`cait.versatile.Line`.
        :type kwargs: Any
        """
        if self._n_channels == 0:
            raise Exception("Nothing to plot.")

        if "xscale" not in kwargs.keys():
            kwargs["xscale"] = "log"
        if "yscale" not in kwargs.keys():
            kwargs["yscale"] = "log"

        if "x" not in kwargs.keys():
            kwargs["x"] = self.freq

        if "xlabel" not in kwargs.keys():
            kwargs["xlabel"] = "Frequency (Hz)"

        if "ylabel" not in kwargs.keys():
            kwargs["ylabel"] = "Noise Power Density (V²/Hz)"

        y = dict()
        for i in range(self._n_channels):
            y[f"NPS Ch-{i}"] = self._ncm[i, i].real
        for i in range(self._n_channels):
            for j in range(i + 1, self._n_channels):
                y[f"abs(CPS ({i},{j}))^2"] = np.abs(self._ncm[i, j]) ** 2

        return Line(y, **kwargs)

    @property
    def _array(self):
        return self._ncm

    @_array.setter
    def _array(self, array):
        if self._ncm.size == 0:
            self._ncm = array
            self._n_ch = array.shape[0]
        else:
            raise Exception("NCM._array can only be set as long as it is empty.")

    @property
    def _n_channels(self):
        return self._n_ch

    @property
    def dt_us(self):
        return self._dt_us

    @property
    def freq(self):
        if self.dt_us is not None:
            n = 2 * (self.shape[-1] - 1)
            return np.fft.rfftfreq(n, self.dt_us / 1e6)
