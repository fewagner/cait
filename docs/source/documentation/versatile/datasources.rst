.. _datasources:

************
Data Sources
************

Data sources are objects that make interfacing the various sources of data as easy as possible. Examples are ``RDTFile`` for interfacing hardware triggered data files, ``Stream`` to access data from a continuous stream file (where it is irrelevant which hardware was used to record the data), but also the object that should be most familiar to ``cait`` users, the ``DataHandler`` is a data source. Data sources do not have to be files but can also provide simulated data on the fly, like ``MockData`` which returns mock voltage traces for quickly testing functions.

The crucial thing is that the technical details on how to access the data behind a data source are hidden from the user. The objects of most interest are voltage traces ("events") and data sources make it easy to access those voltage traces, no matter how they are stored. A data source provides :ref:`event iterators <eventiterators>`.

Top level classes
~~~~~~~~~~~~~~~~~
.. automodule:: cait.versatile
   :members: RDTFile, Stream, MockData
   :member-order: bysource

Related classes and base classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: cait.versatile.datasources.hardwaretriggered.rdt_file
   :members: RDTChannel
.. automodule:: cait.versatile.datasources.stream.streambase
   :members: StreamBaseClass
   