cait.versatile - the flexible cait
==================================

While ``cait`` provides excellent methods for raw data analysis, it is a very rigid framework mostly developed with the needs of the `CRESST` experiment in mind, and if one needs anything out of the ordinary, individualizing the workflow can be cumbersome. The sub-package ``cait.versatile`` aims to streamline this process and provide clear entry points to the existing framework. Moreover, it introduces convenience features for as-fast-as-possible data quality assessment.

The philosophy of ``cait.versatile`` rests upon the following building blocks:

*  **data sources** (:ref:`docs <datasources>`)
    Objects that make interfacing the various sources of data as easy as possible. Examples are ``RDTFile`` for interfacing hardware triggered data files, ``Stream`` to access data from a continuous stream file (where it is irrelevant which hardware was used to record the data), but also the object that should be most familiar to ``cait`` users, the ``DataHandler`` is a data source. Data sources do not have to be files but can also provide simulated data on the fly, like ``MockData`` which returns mock voltage traces for quickly testing functions.

    The crucial thing is that the technical details on how to access the data behind a data source are hidden from the user. The objects of most interest are voltage traces ("events") and data sources make it easy to access those voltage traces, no matter how they are stored. A data source provides **event iterators**.

    **Example:**
    ::
        f = vai.RDTFile("path/to/file.rdt") # construct RDT object
        channels = f[(26,27)]               # choose channels in file
        it = channels.get_event_iterator()  # get event iterator

*  **event iterators** (:ref:`docs <eventiterators>`)
    Objects that represent voltage traces. They are provided by **data sources** and are returned *on-demand*, meaning that it is memory efficient. Event iterators also have properties which tell you how they were recorded. E.g., ``dt_us`` (the time base in microseconds), ``record_length`` (the record length in samples), ``timestamps`` (the microsecond timestamps of the events). 

    Event iterators can be sliced (choose a single channel from a multi channel iterator, or choose a subset of events in the iterator using indices or a boolean flag), added (stick together the events of multiple ``Stream`` and process them in one go, for example) and you can apply **functions** to them.

    Upon constructing an iterator from a data source, you can also specify a ``batch_size``. If it is greater than one, multiple events are read from the data source and returned at once. This reduces the number of file accesses which might speed up your analysis.

    You can tell iterators to apply a processing function on its events before it returns them. Processing functions are discussed below.

    **Example:**
    ::
        it = vai.MockData().get_event_iterator() # quick way to get iterator of two-channel-events for testing
        it_firstch = it[0]                       # new iterator with first channel only
        it_first10 = it[:,:10]                   # new iterator with only first 10 events (all channels)

        # chain event iterators
        it2 = vai.MockData().get_event_iterator()
        combined_it = it + it2

        # Add baseline removal to the iterator. Every time the iterator returns
        # an event, it will have its baseline removed.
        without_bl = it.with_processing(vai.RemoveBaseline())

        # view event traces of an iterator
        vai.Preview(without_bl)

    .. image:: versatile/media/iterator_viewer.png

*  **functions** (:ref:`docs <functions>`)
    Functions can be applied to events or **event iterators** and range from straight-forward ones like ``RemoveBaseline`` (which removes the baselines from given events) to more complex ones like ``OptimumFiltering``. The functions of ``cait.versatile`` are implemented as classes and they can be called like regular functions after construction. The idea behind this is that you specify the parameters of a function at the time of instantiation (e.g. you provide the optimum filter to be used to ``OptimumFiltering`` or the baseline model to ``RemoveBaseline``). After that, the function is just repeatedly called with events for which it should be evaluated. 

    ``cait.versatile`` provides a routine (``apply``) which applies a function to an **event iterator** with built-in options for multi-threading and processing events in batches.

    Functions also expose a ``.preview()`` method which can be used to interactively investigate the effects a function has on events. See below.

    **Example:**
    ::
        it = vai.MockData().get_event_iterator().with_processing(vai.RemoveBaseline())

        # Check the effect the CalcMP function has on the events of the first channel
        # (it plots the event, the moving average that is applied, as well
        # as the points selected to infer time constants and pulse height)
        vai.Preview(it[0], vai.CalcMP())

        # Calculate main parameters by applying CalcMP to the iterator
        pulse_height, onset, rise_time, decay_time, slope = vai.apply(vai.CalcMP(dt_us=it.dt_us))

    .. image:: versatile/media/CalcMP_preview.png

*  **analysis objects** (:ref:`docs <analysisobjects>`)
    Central parts of almost all analyses are the standard event (SEV), the noise power spectrum (NPS) and the optimum filter (OF), which is why we provide dedicated objects ``SEV``, ``NPS``, and ``OF`` to easily use, view and share them. All three objects have classmethods ``from_dh`` and ``from_file`` which let you read them from a ``DataHandler`` or xy-file. The reverse methods ``to_dh`` and ``to_file`` also exist. Furthermore, all three have a method ``show`` which plots the object.
    ``SEV`` and ``NPS`` can also be created from event iterators (e.g. after creating a clean subset of noise traces, you could just hand the iterator to ``NPS`` which will build the noise power spectrum for you).

    **Example:**
    ::
        it = vai.MockData().get_event_iterator().with_processing(vai.RemoveBaseline())

        # Create standard event from events in iterator
        # (usually, you would clean it first)
        sev = vai.SEV(it)

        # Plot the standard event (optional: extract microsecond timestamp from iterator to also show the correct time axis)
        sev.show(dt_us=it.dt_us)   

*  **plotting** (:ref:`docs <plotting>`)
    Often needed plotting tasks are simplified by classes like ``Line``, ``Scatter`` and ``Histogram`` which make having a quick look at your data simple. More sophisticated classes include ``StreamViewer``, which lets you interactively view the contents of the data source ``Stream``, and ``Preview``, which shows a preview of the application of a function to an event iterator. Alternatively, the latter can just be used to view the events in an iterator. 

    The plotting classes are built upon different backends and you can choose the backend using the ``backend`` keyword argument when creating a plot. This lets you, e.g., get an interactive ``plotly`` graph at first, which you can then switch to a ``matplotlib`` graph for your presentations easily.

Detailed Documentations
~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   versatile/datasources
   versatile/iterators
   versatile/functions
   versatile/analysis_objects
   versatile/plotting
   