import os
import mmap
from contextlib import nullcontext

import numpy as np
import h5py

class BinaryFile:
    def __init__(self, path: str, dtype: np.dtype, offset: int = 0, count: int = -1):
        self._mode = "dcache" if path.startswith("dcap://") else "local"
        
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
        if self._mode == "dcache":
            # yes, I tried to use numpy.memmap (uses mmap internally)
            # yes, I tried to use mmap directly (seems not to work in principle)
            # yes, the HDF5 file approach is stupid
            # yes, I am so over dcache
            self._source = h5py.File("proxy", "a", driver="core", backing_store=False)
            g = self._source.require_group("datagroup")
            g.create_dataset("dataset", 
                             shape=(self._size,), 
                             dtype=self._dtype, 
                             external=((self._path, self._offset, self._size*self._dtype.itemsize),))
            
            self._f = self._source["datagroup/dataset"]
            self._isopen = True
            return self._f
        
        else:
            # See the numpy.memmap implementation 
            # (https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/memmap.py)
            start = self._offset - self._offset % mmap.ALLOCATIONGRANULARITY
            array_offset = self._offset - start
            length = int(self._offset + self._size*self._dtype.itemsize) - start

            with open(self._path, "rb") as f:
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
            return self._f
        
    def __exit__(self, typ, val, tb):
            self._source.close()
            self._f = None
            self._isopen = False

    def __getitem__(self, val):
        with self if not self._isopen else nullcontext(self._f) as f:
            return f.__getitem__(val)