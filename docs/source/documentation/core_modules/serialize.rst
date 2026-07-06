**************
cait.serialize
**************

`Serialization <https://en.wikipedia.org/wiki/Serialization>`_ is the process of saving a Python object in a way such that it can be uniquely reconstructed again at a later point. This is relevant for example when you want to save the ``EnergyCalibration`` object to a file or store an ``EventIterator`` in a ``DataHandler`` as a reference. The reverse is called de-serialization and it happens, e.g., when you do ``dh.get_event_iterator()``: The ``DataHandler`` stores a string that uniquely defines the ``EventIterator`` and when you request it, the Python object is constructed again through the process of de-serialization.

The process of serializing is called *dump*, and *load* is the reverse. Serializing into a string specifically is done using *dumps*, and the reverse is *loads*.

.. automodule:: cait.serialize
   :members: SerializingMixin, dump, dumps, load, loads
   :show-inheritance: