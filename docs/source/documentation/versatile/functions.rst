.. _functions:

*********
Functions
*********

There are three different kinds of functions: 

*  `Processing functions`_ take an event and return a processed event (e.g. ``RemoveBaseline``), and they are meant to be used on :ref:`event iterators <eventiterators>`:

.. code-block:: python

   it = vai.MockData().get_event_iterator().with_processing(vai.RemoveBaseline())

*  `Scalar functions`_ take an event and return a scalar (e.g. ``CalcMP`` returns main parameters). They are meant to be applied to entire iterators using the ``apply`` function:

.. code-block:: python
      
   it = vai.MockData().get_event_iterator().with_processing(vai.RemoveBaseline())
   pulse_height, onset, rise_time, decay_time, slope = vai.apply(vai.CalcMP(dt_us=it.dt_us), it)

*  `Utility functions`_ don't fall into those two categories (e.g. ``apply`` or triggering functions).

Processing functions
~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: cait.versatile

.. autoclass:: Downsample
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: RemoveBaseline
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: BoxCarSmoothing
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: TukeyFiltering
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: OptimumFiltering
   :member-order: bysource
   :exclude-members: batch_support

Scalar functions
~~~~~~~~~~~~~~~~
.. currentmodule:: cait.versatile

.. autoclass:: FitBaseline
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: CalcMP
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: TemplateFit
   :member-order: bysource
   :exclude-members: batch_support
.. autoclass:: NPeaks
   :member-order: bysource
   :exclude-members: batch_support

Utility functions
~~~~~~~~~~~~~~~~~
.. currentmodule:: cait.versatile

.. autofunction:: apply
.. autofunction:: trigger_of
.. autofunction:: trigger_zscore

.. currentmodule:: cait.versatile.functions.trigger.triggerbase
.. autofunction:: trigger_base

.. currentmodule:: cait.versatile
.. autofunction:: timestamp_coincidence
.. autofunction:: sample_noise