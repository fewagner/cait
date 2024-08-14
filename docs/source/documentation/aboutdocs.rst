***********************
About the Documentation
***********************

This is the documentation of all functions and classes which are part of ``cait``. It is structured as follows:

- The :ref:`DataHandler class <thedatahandlerclass>` provides convenience functionality for the entire analysis workflow, which also handles storing the data in HDF5 files. Examples are the calculation of main parameters (``dh.calc_mp``) or detector stability (``dh.calc_controlpulse_stability``). 
- If users need any of ``cait``'s functionalities in isolation, i.e. without acting on the HDF5 file underlying the ``DataHandler`` class, directly using functions/classes from the :ref:`Core Modules <coremodules>` is possible. However, most of these functions are wrapped by ``DataHandler`` functions and having to use them directly should be the exception. An example usage would be the following: If the user wants to perform a fit of a standard event themselves, e.g. using ``scipy.optimize.curve_fit``, they can get the pulse shape template directly from ``cait.fit.pulse_template``.
- Finally, if the user needs maximal control over their analysis methods, if they want to prototype new functions, or if they need to access hardware-triggered/stream files directly (without going through the ``DataHandler``), :ref:`cait.versatile <caitversatile>` is the option of choice. Another prominent use-case of this module is plotting: ``cait.versatile`` provides many convenient plotting routines which will speed up your analysis.