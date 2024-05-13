import os
from ctypes import cdll

class TextFile:
    """
    A class that can be used to open text files (e.g. par files) which also supports reading from dcache.

    :param path: The full path (including file extension) to the file of interest. If the path starts with 'dcap://' it is assumed to be a dcache path and reading therefrom is attempted.
    :type path: str

    **Example:**
    ::
        from cait.readers import TextFile

        par_file = "path/to/file.par"

        with TextFile(par_file) as f:
            print(f.read())
    """
    def __init__(self, path: str):
        # Distinguish between reading from dcache or local
        self._mode = "dcache" if path.startswith("dcap://") else "local"
        
        # Check if environment variable is set correctly to use libpdcap.so
        if self._mode == "dcache" and "LD_PRELOAD" not in os.environ.keys():
            raise OSError("To read files from dcache, the environment variable 'LD_PRELOAD' has to be set.")
        if self._mode == "dcache" and not os.path.exists(os.environ["LD_PRELOAD"]):
            raise FileNotFoundError(f"The dcache library does not exist at the specified path {os.environ['LD_PRELOAD']}")
        
        # The differences between reading from dcache or not are:
        # 1. dcache needs a raw string as file path as well as 2. an opener
        # If we wrap these differences in arguments for later, we can just use open() and
        # conveniently get all functionality like readlines() for both implementations
        if self._mode == "dcache":
            self._open_args = ( path if isinstance(path, bytes) else path.encode('utf-8'), "r" )
            self._open_kwargs = { "opener": lambda fname, flags: cdll.LoadLibrary(os.environ["LD_PRELOAD"]).dc_open(fname, 0) }
        
        else:
            self._open_args, self._open_kwargs = ( path, "r" ), {}

    def __enter__(self):
        # return the open file to be used in a context
        self._f = open(*self._open_args, **self._open_kwargs)
        return self._f
    
    def __exit__(self, typ, val, tb):
        self._f.close()