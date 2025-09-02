.. _analysisobjects:

****************
Analysis Objects
****************
The classes listed on this page are solutions for reoccurring tasks in an analysis considered more or less 'standalone' from the rest of ``cait``. E.g. a standard event (SEV) can exist as an array in isolation (it doesn't need a :class:`cait.DataHandler` to be created and it doesn't have to be saved in one). Likewise, to perform an energy calibration, only arrays of testpulse information are needed (which, of course, can be obtained from a :class:`cait.DataHandler`). All the classes below have 'easy-plotting' built in, i.e. it is very easy to assess the quality of the results, or how the methods work in the first place. Eventually, you will probably save the results into a file or back to a :class:`cait.DataHandler`, but the point here is that these objects provide you with a playground for trying out things without commitment until you're happy with the result.

SEV, NPS, NCM, OF
~~~~~~~~~~~~~~~~~
All these objects are 'glorified' ``numpy``-arrays. As such, they can be reshaped, combined to form new arrays, and they carry all typical attributes like ``.shape``, ``.ndim``, ...

.. currentmodule:: cait.versatile

.. autoclass:: SEV
   :members:
   :show-inheritance:
.. autoclass:: NPS
   :members:
   :show-inheritance:
.. autoclass:: NCM
   :members:
   :show-inheritance:
.. autoclass:: OF
   :members:
   :show-inheritance:

Energy Calibration
~~~~~~~~~~~~~~~~~~

.. currentmodule:: cait.versatile

.. autoclass:: EnergyCalibration
   :members:
   :show-inheritance:

Testpulse Responses
-------------------

.. currentmodule:: cait.versatile.analysisobjects.testpulseresponse

.. autoclass:: TestpulseResponse
   :members:
   :show-inheritance:

.. currentmodule:: cait.versatile

.. autoclass:: TPRCubicSpline
   :show-inheritance:
.. autoclass:: TPRPoly
   :show-inheritance:

Transfer Functions
------------------

.. currentmodule:: cait.versatile.analysisobjects.transferfunction

.. autoclass:: TransferFunction
   :members:
   :show-inheritance:

.. currentmodule:: cait.versatile

.. autoclass:: TFPchip
   :show-inheritance:
.. autoclass:: TFPoly
   :show-inheritance: