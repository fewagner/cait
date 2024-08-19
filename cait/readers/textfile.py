import os
import json
from io import StringIO
from ctypes import cdll
from urllib.parse import urlparse

from webdav4.client import Client as WebdavClient
from webdav4.stream import IterStream

try:
    from XRootD import client as RootClient
    from XRootD.client.flags import OpenFlags
    xrootd_installed = True
except ImportError:
    xrootd_installed = False

def get_webdav_file(**kwargs):
    fpath = kwargs.pop("file")
    with IterStream(WebdavClient(**kwargs), fpath) as tf:
        return StringIO(tf.read().decode("ascii"))
    
def get_root_file(**kwargs):
    fpath = kwargs.pop("file")
    with RootClient.File() as f:
        status, _ = f.open(fpath, OpenFlags.READ)
        if status.status: raise IOError(f"Unable to open file using the root protocol: {status}")
        status, data = f.read()
        if status.status: raise IOError(f"Unable to open file using the root protocol: {status}")
    
    return StringIO(data.decode("ascii"))

class TextFile:
    """
    A class that can be used to open text files (e.g. par files) which also supports reading via the Dcap, WebDav, and Root protocol.

    :param path: The full path (including file extension) to the file of interest. If the path starts with ``dcap://``, ``https://`` or ``root://``, reading with the respective protocol is attempted.
    :type path: str
    :type client_kwargs: Any, optional

    The file URL can contain additional arguments used for the request. Example: When using the WebDav protocol, additional keyword arguments for ``webdav4.client.Client`` (https://skshetry.github.io/webdav4/reference/client.html) can be supplied. This can be achieved through URLs like ``https://domain.com/file.txt;{kwarg: value}``.

    **Example:**

    .. code-block:: python

        from cait.readers import TextFile

        par_file = "path/to/file.par"

        with TextFile(par_file) as f:
            print(f.read())
    """
    def __init__(self, path: str):
        # Distinguish between reading from dcache or local
        if path.startswith("dcap://"):
            self._mode = "dcap"
        elif path.startswith("https://"):
            self._mode = "https"
        elif path.startswith("root://"):
            self._mode = "root"
        else:
            self._mode = "local"
        
        # Check if environment variable is set correctly to use libpdcap.so
        if self._mode == "dcap" and "LD_PRELOAD" not in os.environ.keys():
            raise OSError("To read files from dcache, the environment variable 'LD_PRELOAD' has to be set.")
        if self._mode == "dcap" and not os.path.exists(os.environ["LD_PRELOAD"]):
            raise FileNotFoundError(f"The dcache library does not exist at the specified path {os.environ['LD_PRELOAD']}")
        if self._mode == "root" and not xrootd_installed:
            raise FileNotFoundError(f"To access files using the root protocol, the 'xrootd' library has to be installed.")
        
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
            self._open_kwargs = dict(
                file=url.path,
                base_url=f"https://{url.netloc}",
                **(json.loads(url.params) if url.params else {})
            )
            self._opener = get_webdav_file

        elif self._mode == "root":
            self._open_kwargs = { "file": path }
            self._opener = get_root_file
        else:
            self._open_kwargs = { "file": path, "mode": "r" }
            self._opener = open

    def __enter__(self):
        # return the open file to be used in a context
        self._f = self._opener(**self._open_kwargs)
        return self._f
    
    def __exit__(self, typ, val, tb):
        self._f.close()