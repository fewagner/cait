cait.versatile - the hackable cait
==================================

While ``cait`` provides excellent methods for raw data analysis, it is a very rigid framework mostly developed with the needs of the `CRESST` experiment in mind, and if one needs anything out of the ordinary, individualizing the workflow can be cumbersome. The sub-package ``cait.versatile`` aims to streamline this process and provide clear entry points to the existing framework. Moreover, it introduces convenience features for as-fast-as-possible data quality assessment.

The philosophy of ``cait.versatile`` rests upon the following building blocks

*  **data sources**
    Objects that make interfacing the various sources of data as easy as possible. Examples are ``cait.versatile.RDTFile`` for interfacing `.rdt` files, ``cait.versatile.Stream`` to access data from a continuous stream file (where it is irrelevant which hardware was used to record the data), but also the object that should be most familiar to ``cait`` users, the ``cait.DataHandler`` is a data source. Data sources do not have to be files but can also provide simulated data on the fly, like ``cait.versatile.EBPulseSim`` which is an object which overlays simulated pulses on empty baselines.

    The crucial thing is that the technical details on how to access the data behind a data source are hidden from the user. The objects of most interest are voltage traces ("events") and data sources make it easy to access those voltage traces, no matter how they are stored. A data source provides **event iterators**.

*  **event iterators**
    Objects that represent voltage traces. They are provided by **data sources** and 

Detailed Documentations
~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   versatile/raw_data_access
   versatile/analysis_objects
   versatile/processing
   versatile/functions
   versatile/plotting