*******************
About the Tutorials
*******************

All tutorials are written in Jupyter Notebooks, which are stored in the folder **cait/docs/tutorials**. After pulling
cait from git, you can execute them directly inside the tutorials folder,
you might just need to create the folder test_data in the same directory by hand. The folder
is however excluded from  Git, so don't worry about messing up the folder structure.

Please execute the tutorial files **in the given order**, they partially depend on data and features that were generated in
some preceding notebook.

.. note::
    **Script Execution**

    If cait is executed within a Python script rather that with IPython (e.g. Jupyter Notebooks), the main routine has to start with:

        if __name__ == '__main__':

    The need for the explicit main routine specification is common for multithreading in Python.

We like to use and recommend a tool like **HDFView** or **VITables** to view the content of the HDF5 files,
that are used by Cait to store data.
