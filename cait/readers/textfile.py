import os
from ctypes import cdll

class TextFile:
    def __init__(self, path: str):
        self._mode = "dcache" if path.startswith("dcap://") else "local"
        
        if self._mode == "dcache" and "LD_PRELOAD" not in os.environ.keys():
            raise OSError("To read files from dcache, the environment variable 'LD_PRELOAD' has to be set.")
        if self._mode == "dcache" and not os.path.exists(os.environ["LD_PRELOAD"]):
            raise FileNotFoundError(f"The dcache library does not exist at the specified path {os.environ['LD_PRELOAD']}")
        
        if self._mode == "dcache":
            self._open_args = ( path if isinstance(path, bytes) else path.encode('utf-8'), "r" )
            self._open_kwargs = { "opener": lambda fname, flags: cdll.LoadLibrary(os.environ["LD_PRELOAD"]).dc_open(fname, 0) }
        
        else:
            self._open_args, self._open_kwargs = ( path, "r" ), {}

    def __enter__(self):
        self._f = open(*self._open_args, **self._open_kwargs)
        return self._f
    
    def __exit__(self, typ, val, tb):
        self._f.close()