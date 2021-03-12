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
==================

A software project with multiple contributors from various levels of experience needs clear and easy structure to ensure long-term usability. For this reason, please consider the following guidelines in every commit:

- Obey the official code guidelines for Python, outlined in the PEP8 Style Guide (www.python.org/dev/peps/pep-0008). Especially, name functions and variables with lowercase letters and underscores ("recoil_energy") and classes with uppercase letters ("LogicalCut"). Use Python-typical syntax wherever possible (e.g. Numpy vectorisation) rather than C-typical (many loops, ...) and of course do not use global variables.

- Always use descriptive names for functions and variables, avoid abbreviations wherever possible. E.g. "recoil_energy" instead of "E_r", "control_pulse" instead of "c_p". We allow for exceptions in some cases ("sev", "nps", ...), especially if the meaning of a variable is uniquely understandable form the context ("for bl in baselines: ...").

- For complicated, algorithmic code blocks, add comments every few lines of code and explain to the reader what happens here.

- Include a doc string to every function and class with full specification of all arguments and clear explanation of the usage and functionality in the restructured text format. If feasible, include a minimal code example. If the function produces plots, include an example of a plot.

- If you make changes in functions that are not downward compatible to previously released versions, write a changelog in the doc string and raise an instructive exception or warning. If the name of a function or method changes, keep the old one until the next but one release and raise a deprecation warning.

We allow for deviations from the guidelines in rare cases and for good reasons. A good reason for deviating from the guidelines is, if it serves the purpose of usability and readability of your code significantly better, than following them. No good reason would be e.g. because it is a lot a work to adapt your code to the guidelines or because you are not used to them.

Master, Development and Feature Branches
=============================================

We keep a stable and released version on master branch at all times, it is protected from merge requests. Another branch typically hosts a develop version, on which we bring together all new features and fixes for the next release.

While the core developers sometimes do small fixes directly on the development branch, the normal good practice for multiple developers is:

- Open an issue for the fix or feature you want to work on.
- Create a branch for the issue (feature branch).
- Solve the issue and create a merge request.
- Merge the merge request (after review).
- Close the issue.
- Delete the branch you created for the issue.

Please keep as close as possible to this procedure. Keep your changes on the feature branch as close to the issue you defined to work on as possible. Merge the feature branch as soon as possible in the development branch, to avoid merge conflicts due to a drifted apart code base.

Tipps for Debugging
===========================

A usefull tool for  debugging code is the library **IPython pdb** (https://pypi.org/project/ipdb/).
This library exports functions to access the IPython debugger, which features tab completion, syntax highlighting, better tracebacks, better introspection with the same interface as the pdb module.

.. code:: console

    $ pip install ipdb

Adding the line

.. code:: python

    import ipdb; ipdb.set_trace()

any where in your code halts the execution and lets insert and execute additional lines.