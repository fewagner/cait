import os
from io import StringIO
from ctypes import cdll
from urllib.parse import urlparse

from webdav4.client import Client
from webdav4.stream import IterStream

def get_webdav_file(**kwargs):
    fpath = kwargs.pop("file")
    with IterStream(Client(**kwargs), fpath) as tf:
        return StringIO(tf.read().decode("ascii"))

class TextFile:
    """
    A class that can be used to open text files (e.g. par files) which also supports reading from dcache.

    :param path: The full path (including file extension) to the file of interest. If the path starts with 'dcap://' or 'https://' it is assumed to be a dcache path and reading therefrom is attempted.
    :type path: str
    :param client_kwargs: Additional keyword arguments for ``webdav4.client.Client`` (https://skshetry.github.io/webdav4/reference/client.html).
    :type client_kwargs: Any, optional

    **Example:**
    ::
        from cait.readers import TextFile

        par_file = "path/to/file.par"

        with TextFile(par_file) as f:
            print(f.read())
    """
    def __init__(self, path: str, **client_kwargs):
        # Distinguish between reading from dcache or local
        if path.startswith("dcap://"):
            self._mode = "dcap"
        elif path.startswith("https://"):
            self._mode = "https"
        else:
            self._mode = "local"
        
        # Check if environment variable is set correctly to use libpdcap.so
        if self._mode == "dcap" and "LD_PRELOAD" not in os.environ.keys():
            raise OSError("To read files from dcache, the environment variable 'LD_PRELOAD' has to be set.")
        if self._mode == "dcap" and not os.path.exists(os.environ["LD_PRELOAD"]):
            raise FileNotFoundError(f"The dcache library does not exist at the specified path {os.environ['LD_PRELOAD']}")
        
        # The differences between reading from dcache or not are:
        # 1. dcache needs a raw string as file path as well as 2. an opener
        # If we wrap these differences in arguments for later, we can just use open() and
        # conveniently get all functionality like readlines() for both implementations
        if self._mode == "dcap":
            self._open_kwargs = { 
                "file": path if isinstance(path, bytes) else path.encode('utf-8'),
                "mode": "r",
                "opener": lambda fname, flags: cdll.LoadLibrary(os.environ["LD_PRELOAD"]).dc_open(fname, 0) 
            }
            self._opener = open
        elif self._mode == "https":
            url = urlparse(path)
            self._open_kwargs = {
                "file": url.path,
                "base_url": f"https://{url.netloc}",
                "verify": client_kwargs.pop("verify", False)
            }
            self._opener = get_webdav_file
        else:
            self._open_kwargs = { "file": path, "mode": "r" }
            self._opener = open

    def __enter__(self):
        # return the open file to be used in a context
        self._f = self._opener(**self._open_kwargs)
        return self._f
    
    def __exit__(self, typ, val, tb):
        self._f.close()