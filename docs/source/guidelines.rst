***********************
Contribution Guidelines
***********************

The repository has the following structure:

.. code::
    .
    +--cait/
    |  +--folder1/
    |    +--__init__py.
    |    +--_utility_functions.py
    |  +--__init__py.
    |  +--MajorClass.py
    +--dist/
    |  +--some_wheel_file.whl
    +--tests/
    |  +--__init__py.
    |  +--some_test.py
    +--docs/
    |  +--source
    |    +--tutorials
    |       +--conversion.ipynb
    |    +--index.rst
    |  +--Makefile
    +--.gitignore
    +--setup.py

The **cait** folder contains all source code for the package. It directly contains the major classes (EventInterface, DataHandler, ...) and several sub-folders for utility functions. The sub-folders are thematically structured (filter, fit, data, ...) and the intention is, that the user should mainly interact with the utility functions through the major classes. However, advanced users can access some of the utility functions directly (such aus generating filters, fits, etc). Every folder contains an __init__.py file, which specifies all functions that are directly accessable for the user (__all__ = [...]).

The **dist** folder contains the wheel files, that are used for installation. These are automatically created when calling

.. code:: console

    $ python setup.py bdist_wheel

in the top level directory.

The **tests** folder contains test files that can be automatically executed.

The **docs** folder contains the documentation, written in restructured text, automatically generated with Sphinx. The tutorials subfolder contains tutorials in Jupyter notebooks, that are automatically rendered into the documentation.

The **.gitignore** file excludes all files that are too large for git or specific for each user: data, virtual environment, ...

The **setup.py** file contains information for the package: dependencies, version number, ...

Code Guidelines
=========

A software project with multiple contributors from various levels of experience needs clear and easy structure to ensure long-term usability. For this reason, please consider the following guidelines in every commit:

- Obey the official code guidelines for Python, outlined in the PEP8 Style Guide.

- Always use descriptive names for functions and variables, avoid abbreviations wherever possible. E.g. "recoil_energy" instead of "er", "control_pulse" instead of "cp". We allow for exceptions in some cases ("sev", "nps", ...), especially if the meaning of a variable is uniquely understandable form the context ("for bl in baselines: ...").

- Include a doc string to every function and class with full specification of all arguments and clear explanation the usage and functionality.

Debugging
=========

A usefull tool for  debugging code is the library [**IPython pdb**](https://pypi.org/project/ipdb/) (_short: ipdb_).
This library exports functions to access the IPython debugger, which features tab completion, syntax highlighting, better tracebacks, better introspection with the same interface as the pdb module.

.. code:: console

    $ pip install ipdb

Adding the line

.. code:: python

    import ipdb; ipdb.set_trace()

any where in your code halts the execution and lets insert and execute additional lines.