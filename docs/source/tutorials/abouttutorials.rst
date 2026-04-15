*******************
About the Tutorials
*******************

All tutorials are written in Jupyter Notebooks, which are stored in the folder **cait/docs/tutorials**. After pulling ``cait`` from GitLab/GitHub, you can execute them directly inside the tutorials folder.

We tried to make the tutorials self-contained, meaning that you don't *have* to execute them in order. Nevertheless, they are conceptually built on top of each other so it only makes sense to go in order. Some of the notebooks do, however, depend on the results of the previous one, in which case you'd have to execute them first (it would have been much more cumbersome to make them self-contained -- please forgive us).

Quick overview
--------------

Recently, all experiments who use ``cait`` have switched to using so-called *stream data*, i.e. continuously recorded detector data that needs to be triggered before doing any other analysis. Therefore, we start with the **Triggering stream data** tutorial. The results of the triggering, like all other intermediate results in ``cait``, are stored in `HDF5 files <https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html>`_ which are in turn wrapped by the ``DataHandler`` class in ``cait``. The second tutorial **Interacting with HDF5 files** walks you through how to use it.
... *More to come*