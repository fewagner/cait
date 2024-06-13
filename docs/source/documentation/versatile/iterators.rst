.. _eventiterators:

***************
Event Iterators
***************

In general, iterators are never created directly by the user but provided by the respective **data source** through the ``get_event_iterator`` method. E.g. ``Stream`` provides a ``StreamIterator`` and ``RDTChannel`` provides an ``RDTIterator``. Irrespective of what the underlying data source is, all iterators inherit from ``IteratorBaseClass`` and share its methods and properties which are documented below.

.. automodule:: cait.versatile.iterators.iteratorbase
   :members: IteratorBaseClass
   :member-order: bysource