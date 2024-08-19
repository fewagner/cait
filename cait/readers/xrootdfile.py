from typing import List, Tuple, Union
from contextlib import nullcontext
from urllib.parse import urlparse

import numpy as np

from XRootD import client
from XRootD.client.flags import OpenFlags, QueryCode

from .helper import sanitize_slice, is_index_list, is_str_like

def get_filesize(f: client.File):
    """docstring"""
    status, stats = f.stat()
    if status.status: raise IOError(f"Cannot determine filesize: {status}")
    
    return stats.size

def get_root_vread_info(server_url: str):
    """docstring"""
    c = client.FileSystem(server_url)
    # maximum number of elements in a kXR_readv request vector
    status, readv_iov_max = c.query(QueryCode.CONFIG, 'readv_iov_max')
    if status.status:
        print("Cannot determine maximum number of elements in a kXR_readv request. Use default (1024).")
        readv_iov_max = 1024
              
    # maximum amount of data that may be requested in a single kXR_readv request element
    status, readv_ior_max = c.query(QueryCode.CONFIG, 'readv_ior_max')
    if status.status:
        print("Cannot determine maximum amount of data that may be requested in a single kXR_readv request. Use default (2097136).")
        readv_ior_max = 2097136
    
    return { "max_elem": int(readv_iov_max), "max_bytes": int(readv_ior_max) }

def divide_chunks(chunks: List[Tuple[int]],
                  max_elem: int = 1024,
                  max_bytes: int = 2097136):
    """docstring"""
    size_current_read = 0
    n_current_read = 0
    current_read = []
    divided_reads = []
    
    for i in range(len(chunks)):
        size_current_read += chunks[i][1]
        n_current_read += 1
        
        if (size_current_read > max_bytes) or (n_current_read > max_elem):
            divided_reads.append(current_read)
            current_read = [chunks[i]]
            size_current_read = chunks[i][1]
            n_current_read = 1
        else:
            current_read.append(chunks[i])
            
        if i == len(chunks)-1: divided_reads.append(current_read)
            
    return divided_reads

def vread(f: client.File,
          dtype: np.dtype,
          chunks: List[Tuple[int]],
          **v_read_info):
    """docstring"""
    list_of_chunks = divide_chunks(chunks, **v_read_info)
    
    list_of_outputs = []
    for chunks in list_of_chunks:
        status, response = f.vector_read(chunks=chunks)
        if status.status: raise IOError(f"Unable to vector_read from file using the root protocol: {status}")
        list_of_outputs.append(list(response.chunks))
    
    # Collect all chunks into one output
    return np.concatenate([np.frombuffer(chunk.buffer, dtype=dtype) for chunks in list_of_outputs for chunk in chunks]) 


def field_read(f: client.File, 
               dtype: np.dtype, 
               name: Union[str, List[str]], 
               offset: int = 0, 
               select: Union[slice, list, int] = None,
               **v_read_info):
    """docstring"""
    if dtype.names is None: raise TypeError("Not a dtype with named fields.")
    names = [name] if isinstance(name, str) else name
    
    n_items = (get_filesize(f)-offset)//dtype.itemsize
    item_inds = np.arange(n_items)
    if select is not None: 
        item_inds = item_inds[select]
        if isinstance(item_inds, (int, np.integer)): 
            item_inds = [item_inds]
    
    all_reads = []
    all_dtypes = []
    
    for name in names:
        all_dtypes.append((name, np.dtype(dtype[name]).str))
        field_ind = dtype.names.index(name)
        field_size = dtype[name].itemsize
        add_offset = sum([dtype[k].itemsize for k in range(field_ind)])
    
        chunks = [(int(offset + add_offset + k*dtype.itemsize), field_size) for k in item_inds]
        
        all_reads.append(vread(f, dtype[name], chunks, **v_read_info))


    return np.array(list(zip(*all_reads)), dtype=np.dtype(all_dtypes)) if len(all_reads)>1 else all_reads[0]

def slice_read(f: client.File, 
               dtype: np.dtype,
               sl: slice,
               offset: int = 0):
    """docstring"""
    status, response = f.read(offset=offset+sl.start, size=(sl.stop-sl.start)*dtype.itemsize)
    if status.status: raise IOError(f"Unable to read from file using the root protocol: {status}")

    data = np.frombuffer(response, dtype=dtype)
    return data[::sl.step] if sl.step > 1 else data

class XRootDReader:
    def __init__(self, url: str, dtype: np.dtype, offset: int = 0, count: int = -1):
        self._fpath = url
        self._fserver = f"{urlparse(url).scheme}://{urlparse(url).hostname}"
        self._dtype = dtype
        self._offset = offset
        
        with self:
            self._flen = get_filesize(self._f) - offset

        # Error handling for inappropriate count arguments is done in BinaryFile
        self._size = self._flen // self._dtype.itemsize if count == -1 else count
        
        self._vread_info = get_root_vread_info(self._fserver)

        self._isopen = False
        
    def __len__(self):
        return self._flen
        
    def __enter__(self):
        self._f = client.File().__enter__()
        status, _ = self._f.open(self._fpath, OpenFlags.READ)
        if status.status: IOError(f"Unable to open file using the root protocol: {status}")
        
        self._isopen = True
        return self
    
    def __exit__(self, typ, val, tb):
        self._f.__exit__(typ, val, tb)
        self._isopen = False
        
    def __getitem__(self, val):
        with self if not self._isopen else nullcontext(self) as s:
            if is_str_like(val):
                return field_read(s._f, self._dtype, val, self._offset, **self._vread_info)
            elif is_index_list(val):
                return vread(s._f, self._dtype, 
                             [(self._offset+i*self._dtype.itemsize, self._dtype.itemsize) for i in val], **self._vread_info)
            elif isinstance(val, (int, np.integer)):
                return self[[val]]
            elif isinstance(val, np.ndarray) and val.ndim<2:
                return self[val.tolist()]
            elif isinstance(val, slice):
                return slice_read(s._f, self._dtype, sanitize_slice(val, self._size), self._offset)
            
            elif isinstance(val, tuple) and len(val) == 2:
                if is_str_like(val[0]):
                    return field_read(s._f, self._dtype, val[0], self._offset, val[1], **self._vread_info)
                elif isinstance(val[0], (int, np.integer, slice)) or is_index_list(val[0]):
                    return self[val[0]][val[1]]
                else:
                    raise NotImplementedError(f"")
            else:
                raise NotImplementedError(f"for indexing arguments {val} of type {[type(v) for v in val] if isinstance(val, tuple) else type(val)}")