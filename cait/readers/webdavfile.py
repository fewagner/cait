from typing import Union, BinaryIO
from urllib.parse import urlparse
from contextlib import nullcontext

import numpy as np
from webdav4.client import Client
from webdav4.stream import IterStream

def unpack_buffer(buffer: bytearray, dtype: np.dtype, select: Union[slice, list, int] = None):
    """Helper function to unpack a binary array according to a given dtype. Additional slicing for the array after unpacking can be specified using the 'select' keyword."""
    arr = np.frombuffer(buffer, dtype=dtype)
    return arr if select is None else arr[select]

def read_index(f: BinaryIO, 
               dtype: np.dtype, 
               offset: int, 
               index: int, 
               select: Union[slice, list, int] = None):
    """Helper function to read and unpack one item in a binary file according to a given dtype. Additional slicing for the array after unpacking can be specified using the 'select' keyword."""
    f.seek(offset+index*dtype.itemsize)
    return unpack_buffer(f.read(dtype.itemsize), dtype, select)

def read_slice(f: BinaryIO, 
               dtype: np.dtype, 
               offset: int, 
               sl: slice, 
               chunksize: int = 1000, 
               select: Union[slice, list, int] = None):
    """Helper function to read and unpack items specified by a slice of a binary file according to a given dtype. Additional slicing for the array after unpacking can be specified using the 'select' keyword."""
    start = sl.start + offset
    stop = sl.stop + offset
    n_items = stop - start
    n_chunks, remainder = n_items//(chunksize), n_items%(chunksize) 
    chunks = [chunksize*dtype.itemsize]*n_chunks + ([remainder*dtype.itemsize] if remainder else [])
    
    f.seek(offset+start*dtype.itemsize)
    return np.concatenate(
            [unpack_buffer(f.read(c), dtype, select) for c in chunks]
        )[slice(None, None, sl.step)]

def read_list(f: BinaryIO, 
              dtype: np.dtype, 
              offset: int, 
              l: list, 
              select: Union[slice, list, int] = None):
    """Helper function to read and unpack items in a binary file according to a given dtype and indices in a list. Additional slicing for the array after unpacking can be specified using the 'select' keyword."""
    out = []
    for v in l:
        f.seek(offset+v*dtype.itemsize)
        out.append(unpack_buffer(f.read(dtype.itemsize), dtype, select))
    return np.concatenate(out)

def sanitize_slice(sl: slice, size: int):
    """Converts None values in slice to appropriate integers."""
    start = sl.start if sl.start else 0
    stop = sl.stop if sl.stop else size
    return slice(start, stop, sl.step)

def is_index_list(val):
    """Returns true if input is a list of integers."""
    return isinstance(val, list) and all([isinstance(x, (int, np.integer)) for x in val])

def is_str_like(val):
    """Returns true if input is a string or a list of strings."""
    return isinstance(val, str) or (isinstance(val, list) and all([isinstance(x, str) for x in val]))

class WebdavReader:
    """
    File reader for dcache files accessed via the WebDAV protocol.

    :param url: The full path (including file extension and the 'https' prefix) to the file of interest.
    :type url: str
    :param dtype: The numpy (structured) dtype to use when interpreting the contents of the file.
    :type dtype: np.dtype
    :param offset: The offset (in bytes) for reading the file, defaults to 0.
    :type offset: int, optional
    :param count: The number of items (of size given by dtype) to read. If -1, the entire file is read, defaults to -1.
    :type count: int, optional
    :param client_kwargs: Additional keyword arguments for ``webdav4.client.Client`` (https://skshetry.github.io/webdav4/reference/client.html).
    :type client_kwargs: Any, optional
    """
    def __init__(self, url: str, dtype: np.dtype, offset: int = 0, count: int = -1, **client_kwargs):
        url = urlparse(url)
        verify = client_kwargs.pop("verify", False)
        self._client = Client(base_url=f"https://{url.netloc}", verify=verify, **client_kwargs)
        self._fpath = url.path
        self._dtype = dtype
        self._offset = offset
        
        with IterStream(self._client, self._fpath) as f:
            f.seek(0, 2)
            self._flen = f.tell() - offset

        # Error handling for inappropriate count arguments is done in BinaryFile
        self._size = self._flen // self._dtype.itemsize if count == -1 else count

        self._isopen = False

    def __len__(self):
        return self._flen
        
    def __enter__(self):
        self._f = IterStream(self._client, self._fpath).__enter__()
        self._isopen = True
        return self
    
    def __exit__(self, typ, val, tb):
        self._f.__exit__(typ, val, tb)
        self._isopen = False
        
    def __getitem__(self, val):
        with self if not self._isopen else nullcontext(self._f) as f:
            if isinstance(val, slice):
                val = sanitize_slice(val, self._size)
                return read_slice(f, self._dtype, self._offset, val)
                
            elif isinstance(val, (int, np.integer)):
                return read_index(f, self._dtype, self._offset, val)

            elif is_str_like(val):
                sl = sanitize_slice(slice(None), self._size)
                return read_slice(f, self._dtype, self._offset, sl, select=val)
            
            elif is_index_list(val):
                return read_list(f, self._dtype, self._offset, val)
            
            elif isinstance(val, tuple) and len(val) == 2:
                if isinstance(val[0], slice):
                    sl = sanitize_slice(val[0], self._size)
                    return read_slice(f, self._dtype, self._offset, sl, select=val[1])
                elif isinstance(val[0], (int, np.integer)):
                    return read_index(f, self._dtype, self._offset, val[0], select=val[1])
                elif is_str_like(val[0]):
                    if is_str_like(val[1]):
                        raise NotImplementedError(f"for indexing arguments of type {[type(v) for v in val]}")
                    return self[val[1], val[0]]
                elif is_index_list(val[0]):
                    return read_list(f, self._dtype, self._offset, val[0], select=val[1])
                else:
                    raise NotImplementedError(f"for indexing arguments of type {[type(v) for v in val]}")
            else:
                raise NotImplementedError(f"for indexing arguments {val}")
