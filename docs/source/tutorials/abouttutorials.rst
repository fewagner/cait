*******************
About the Tutorials
*******************

All tutorials are written in Jupyter Notebooks, which are stored in the folder **cait/docs/tutorials**. After pulling ``cait`` from GitLab/GitHub, you can execute them directly inside the tutorials folder.

We tried to make the tutorials self-contained, meaning that you don't *have* to execute them in order. Nevertheless, they are conceptually built on top of each other so it only makes sense to go in order.

Note that here we present the analysis from a ``cait`` usage perspective. Our goal is not to teach you *why* we do things or what the methods we apply are in detail. If you want to dive deeper into raw data analysis, we recommend to read the theses of previous students in the COSINUS, CRESST, and NUCLEUS experiment. Many of them have detailed sections on raw data analysis. You'll find them here:

- https://www.cosinus.it/dissertations/
- https://cresst-experiment.org/publications/phd-theses
- https://nucleus-experiment.org/publications

Quick overview
--------------

Recently, all experiments who use ``cait`` have switched to using so-called *stream data*, i.e. continuously recorded detector data that needs to be triggered before doing any other analysis. Therefore, we start with the **Triggering stream data** tutorial. The results of the triggering, like all other intermediate results in ``cait``, are stored in `HDF5 files <https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html>`_ which are in turn wrapped by the ``DataHandler`` class in ``cait``. The second tutorial 
, **Interacting with HDF5 files**, walks you through how to use it.

In **Creating SEV, NPS, OF**, we show you how pulse templates or standard events (SEVs), noise power spectra (NPSs), and optimum filters (OFs) are constructed. They are central to any analysis and we show you several ways how you can create them. Here, we also teach you how to perform quality cuts to the data and how you can estimate the baseline resolution of your detector.

With the objects from before at hand, we will outline the two main approaches to obtain a pulse height spectrum in **Reconstructing the pulse amplitude**. Converting those pulse heights to energies while taking the detector response over time into account, is subject of **Energy Calibration**.

The energy spectrum only tells the full story once you know how well your detector can detect signals. In **Efficiency Simulation**, you'll learn now you can calculate an energy-dependent efficiency.

All of the above will be done with minimal examples. Actual analyses often require you to analyze several tens or hundreds of files. The tutorial **Processing many files using SLURM jobs** shows you how you can easily scale your analysis.

... *More to come*

.. note::
    The remaining tutorials that you currently find in the navigation tree are leftovers from previous ``cait`` versions. They are not actively maintained at the moment, unfortunately. But we are working on it! Just know that the contents of those notebooks might not reflect the current best practices to do analysis in ``cait`` and that you might encounter errors due to compatibility issues with ``cait``'s dependencies.