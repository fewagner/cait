***************************************
cait.readers
***************************************

We provide readers ``TextFile`` for text file such as ``.par``-files, and ``BinaryFile`` for binary data files of arbitrary data type. These two classes are wrappers that automatically handle local and remote files: To access files on a locally mounted drive, just use ``regular/file/paths.txt``. If you prefix file paths with ``<protocol>://server.data.at/``, the file is read from remote. Currently, we support the ``dcap``, ``https`` (WebDav), and ``root`` (XRootD) protocol and we highly recommend using ``root`` wherever possible.

.. automodule:: cait.readers
   :members: TextFile, BinaryFile
   :show-inheritance: