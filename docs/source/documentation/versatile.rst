cait.versatile - the hackable cait
==================================

While ``cait`` provides excellent methods for raw data analysis, it is a very rigid framework mostly developed with the needs of the `CRESST` experiment in mind, and if one needs anything out of the ordinary, individualizing the workflow can be cumbersome. The sub-package ``cait.versatile`` aims to streamline this process and provide clear entry points to the existing framework. Moreover, it introduces convenience features for as-fast-as-possible data quality assessment.

The philosophy of ``cait.versatile`` rests upon the following building blocks:

*  **data sources**
    Objects that make interfacing the various sources of data as easy as possible. Examples are ``RDTFile`` for interfacing hardware triggered data files, ``Stream`` to access data from a continuous stream file (where it is irrelevant which hardware was used to record the data), but also the object that should be most familiar to ``cait`` users, the ``DataHandler`` is a data source. Data sources do not have to be files but can also provide simulated data on the fly, like ``EBPulseSim`` which is an object which overlays simulated pulses on empty baselines, or ``MockData`` which returns mock voltage traces for quickly testing functions.

    The crucial thing is that the technical details on how to access the data behind a data source are hidden from the user. The objects of most interest are voltage traces ("events") and data sources make it easy to access those voltage traces, no matter how they are stored. A data source provides **event iterators**.

*  **event iterators**
    Objects that represent voltage traces. They are provided by **data sources** and are returned *on-demand*, meaning that it is memory efficient. Event iterators have properties which tell you about how they were recorded. E.g., ``dt_us`` (the time base in microseconds), ``record_length`` (the record length in samples), ``timestamps`` (the microsecond timestamps of the events). 

    Event iterators can be sliced (choose a single channel from a multi channel iterator, or choose a subset of events in the iterator using indices or a boolean flag), added (stick together the events of multiple ``Stream``s and process them in one go, for example) and you can apply **functions** to them.

*  **functions**
    Functions can be applied to events or **event iterators** and range from straight-forward ones like ``RemoveBaseline`` (which removes the baselines from given events) to more complex ones like ``OptimumFiltering``. The functions of ``cait.versatile`` are implemented as classes and they can be called like regular functions after construction. The idea behind this is that you specify the parameters of a function at the time of instantiation (e.g. you provide the optimum filter to be used to ``OptimumFiltering`` or the baseline model to ``RemoveBaseline``). After that, the function is just repeatedly called with events for which it should be evaluated. 

    ``cait.versatile`` provides a routine (``apply``) which applies a function to an **event iterator** with built-in options for multi-threading and reading/processing events in batches.

    Functions also expose a ``.preview()`` method which can be used to interactively investigate the effects a function has on events. See below.

*  **analysis objects**
    

*  **plotting**
    Often needed plotting tasks are simplified by classes like ``Line``, ``Scatter`` and ``Histogram`` which make having a quick look at your data simple. More sophisticated classes include ``StreamViewer``, which lets you interactively view the contents of the **data source** ``Stream``, and ``Preview``, which shows a preview of the application of a **function** to an **event iterator**. Alternatively, the latter can just be used to view the events in an iterator. 

    The plotting classes are built upon different backends and you can choose the backend using the ``backend`` keyword argument when creating a plot. This lets you, e.g., get an interactive ``plotly`` graph at first, which you can then switch to a ``matplotlib`` graph for your presentations easily.

Detailed Documentations
~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   versatile/raw_data_access
   versatile/analysis_objects
   versatile/processing
   versatile/functions
   versatile/plotting