.. _eventiterators:

***************
Event Iterators
***************

In general, iterators are never created directly by the user but provided by the respective **data source** through the ``get_event_iterator`` method. E.g. ``Stream`` provides a ``StreamIterator`` and ``RDTChannel`` provides an ``RDTIterator``. Irrespective of what the underlying data source is, all iterators inherit from ``IteratorBaseClass`` and share its methods and properties which are documented below.

In particular, you can view the events in an iterator ``it`` by calling ``vai.Preview(it)``, add processing (functions that are applied to each event in an iterator before it is returned) through ``it.with_processing(vai.RemoveBaseline())`` or ``it.with_processing([vai.RemoveBaseline(), lambda x: x**2])``, etc., and request the iterator's properties, e.g. ``it.t`` (the time array of the events), ``it.timestamps`` (timestamps of events in iterator), ``it.dt_us`` (microsecond timebase used to record the events in the iterator), ...

As mentioned, iterators are usually obtained using the ``get_event_iterator`` method (e.g. ``dh.get_event_iterator('events')``). One iterator that *is* useful to explicitly create yourself, however, is the :class:`cait.versatile.iterators.PulseSimIterator` which can be used to simulate pulses given some SEV (or template parameters), pulse heights and shifts (see below). 

.. autoclass:: cait.versatile.iterators.iteratorbase.IteratorBaseClass
   :members: 
   :member-order: bysource

.. autoclass:: cait.versatile.iterators.PulseSimIterator
   