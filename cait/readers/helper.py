from typing import Union

import numpy as np

def unpack_buffer(buffer: bytearray, dtype: np.dtype, select: Union[slice, list, int] = None):
    """Helper function to unpack a binary array according to a given dtype. Additional slicing for the array after unpacking can be specified using the 'select' keyword."""
    arr = np.frombuffer(buffer, dtype=dtype)
    return arr if select is None else arr[select]

def sanitize_slice(sl: slice, size: int):
    """Converts None values in slice to appropriate integers."""
    start = sl.start if sl.start else 0
    stop = sl.stop if sl.stop else size
    step = sl.step if sl.step else 1
    return slice(start, stop, step)

def sanitized_dtype(dtype: np.dtype):
    """
    Removes empty items (of itemsize 0) from the dtype. This is used, e.g., for HDF5 files which cannot handle size 0 datasets. Notice that removing them does not alter how the files are read.

    :param dtype: dtype to be sanitized
    :type dtype: np.dtype

    :return: Same dtype but with items of size 0 removed
    :rtype: np.dtype
    """
    if dtype.fields:
        # Structured dtype
        out = np.dtype({k:v for k,v in dtype.fields.items() if dtype[k].itemsize > 0})
    else:
        # Plain dtype
        out = dtype if dtype.itemsize > 0 else np.dtype({}) 

    if out.itemsize == 0: raise ValueError("The sanitized dtype has itemsize 0.")

    return out

def is_index_list(val):
    """Returns true if input is a list of integers."""
    return isinstance(val, list) and all([isinstance(x, (int, np.integer)) for x in val])

def is_str_like(val):
    """Returns true if input is a string or a list of strings."""
    return isinstance(val, str) or (isinstance(val, list) and all([isinstance(x, str) for x in val]))