import os
import mmap
from contextlib import nullcontext

import numpy as np
import h5py

class BinaryFile:
    """
    A class that can be used to open binary files (e.g. rdt or stream files) which also supports reading from dcache.

    :param path: The full path (including file extension) to the file of interest. If the path starts with 'dcap://' it is assumed to be a dcache path and reading therefrom is attempted.
    :type path: str
    :param dtype: The numpy (structured) dtype to use when interpreting the contents of the file.
    :type dtype: np.dtype
    :param offset: The offset (in bytes) for reading the file, defaults to 0.
    :type offset: int, optional
    :param count: The number of items (of size given by dtype) to read. If -1, the entire file is read, defaults to -1.
    :type count: int, optional

    **Example:**
    ::
        import numpy as np
        from cait.readers import BinaryFile
        from cait.versatile import Line

        rdt_file = "path/to/file.rdt"
        dtype = np.dtype([ ('detector_nmbr', 'i4'), ('coincide_pulses', 'i4'),
                           ('trig_count', 'i4'), ('trig_delay', 'i4'),
                           ('abs_time_s', 'i4'), ('abs_time_mus', 'i4'),
                           ('delay_ch_tp', 'i4', (1,)), ('time_low', 'i4'),
                           ('time_high', 'i4'), ('qcd_events', 'i4'),
                           ('hours', 'f4'), ('dead_time', 'f4'),
                           ('test_pulse_amplitude', 'f4'), ('dac_output', 'f4'),
                           ('samples', 'i2', 16384),
                          ])

        with BinaryFile(rdt_file, dtype=dtype) as f:
            first_event = np.array(f[0]["samples"])

        Line(first_event)
    """
    def __init__(self, path: str, dtype: np.dtype, offset: int = 0, count: int = -1):
        # Distinguish between reading from dcache or local
        self._mode = "dcache" if path.startswith("dcap://") else "local"
        
        # Check if environment variable is set correctly to use libpdcap.so
        if self._mode == "dcache" and "LD_PRELOAD" not in os.environ.keys():
            raise OSError("To read files from dcache, the environment variable 'LD_PRELOAD' has to be set.")
        if self._mode == "dcache" and not os.path.exists(os.environ["LD_PRELOAD"]):
            raise FileNotFoundError(f"The dcache library does not exist at the specified path {os.environ['LD_PRELOAD']}")
        
        flen = os.path.getsize(path) - offset
        itemsize = dtype.itemsize

        if count == -1:
            if flen % itemsize:
                raise ValueError(f"File size ({flen}) is not a multiple of dtype size ({itemsize}).")
            
            self._size =  flen // itemsize
        else:
            if count > flen // itemsize:
                raise ValueError(f"Count ({count}) exceeds number of items in file ({flen//itemsize}).")
            
            self._size = count
        
        self._path = path
        self._dtype = dtype
        self._offset = offset
        self._count = count

        self._f = None
        self._isopen = False

    def __enter__(self):
        # yes, I tried to use numpy.memmap (uses mmap internally)
        # yes, I tried to use mmap directly (seems not to work in principle)
        # yes, the HDF5 file approach is stupid
        # yes, I am so over dcache
        if self._mode == "dcache":
            # Create h5 dummy file in memory (driver="core" does not create a file
            # and backing_store=False prevents generating an output file once closed)
            self._source = h5py.File("proxy", "a", driver="core", backing_store=False)
            g = self._source.require_group("datagroup")
            g.create_dataset("dataset", 
                             shape=(self._size,), 
                             dtype=self._dtype, 
                             external=((self._path, self._offset, self._size*self._dtype.itemsize),))
            
            # The datasets of h5py are quite similar to numpy arrays. In particular,
            # they can be indexed identically. Therefore, returning the h5 dataset here
            # and the numpy memmap else creates an identical data handling
            self._f = self._source["datagroup/dataset"]
            self._isopen = True
            return self._f
        
        # See the numpy.memmap implementation for reference
        # (https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/memmap.py)
        else:
            start = self._offset - self._offset % mmap.ALLOCATIONGRANULARITY
            array_offset = self._offset - start
            length = int(self._offset + self._size*self._dtype.itemsize) - start

            with open(self._path, "rb") as f:
                # create the mmap object separately so that it can be nicely closed
                # (numpy memmap does not close the mmap object!)
                self._source = mmap.mmap(fileno=f.fileno(),
                                         length=length,
                                         access=mmap.ACCESS_READ,
                                         offset=start)
            self._isopen = True
            self._f = np.ndarray.__new__(np.ndarray,
                                         shape=(self._size,),
                                         dtype=self._dtype,
                                         buffer=self._source,
                                         offset=array_offset)
            # The array returned here is treated identical to the h5 dataset returned
            # in case we are reading from dcache.
            return self._f
        
    def __exit__(self, typ, val, tb):
            self._source.close()
            self._f = None
            self._isopen = False

    def __getitem__(self, val):
        with self if not self._isopen else nullcontext(self._f) as f:
            # When entering the context, self.__enter__ returns f, which
            # we made sure behaves the same when being indexed in both cases. 
            # Hence, we can just forward the __getitem__ call to the
            # (numpy or hdf5) array
            return f.__getitem__(val)