************
Installation
************

There are various ways for the installation of cait, which are outlined in the following. Usually we provide a stable
release, typically hosted on the GitLab/GitHub master branch and a development version. The development version might
still be unstable and undocumented in some new features, does however include all cutting-edge and current implementations.
Our recommendation is therefore to use the development branch, combined with active bug-reporting in the GitLab/Hub
issue tracker. Due to the still very small user and developer community, the stable release is also to be understood as
a beta version.

**Important Note for JupyterHub on computing clusters:**

In the past, many users experienced issues with our interactive plotting tools which are based on plotly and ipywidgets. These problems were due to version mismatches between the plotly/ipywidgets packages and their corresponding JupyterLab extensions (which are automatically installed alongside the packages). 

To not run into such issues in the first place, we recommend one of the following approaches:

* Install cait in the base environment of your computing cluster's JupyterLab. 
* Install it in a virtual environment (because you want or have to) is also possible but you will have to make sure that the same plotly/ipywidgets versions (which are installed as dependencies of cait) are also present in the base environment. A good practice to ensure this is to always pip-upgrade plotly/ipywidgets to the latest version in both environments. Note that you will potentially have to match these versions every time you upgrade cait.

Lastly, remember to **restart JupyterHub completely** (not just the kernel) for the changes to take effect.

Installation from PyPI (recommended)
====================================

Cait is hosted on the Python package index.

.. code:: console

    $ pip install cait

For older or unreleased version, use the installation from Git.

Installation from Wheel File
====================================

For installing Cait, first install and upgrade the following helper libraries:

.. code:: console

    $ pip install -U wheel setuptools twine

Then clone the Git repository. The folder of the repository contains a wheel file:

.. code:: console

    $ dist/*.whl

If there are multiple wheel files, choose the one with the highes version number.
For installation of the library, run:

.. code:: console

    $ pip install /path/to/wheelfile.whl

You can now import the library in Python, e.g.

.. code:: python

    import cait as ai
    from cait import EventInterface

Options for Developers
====================================

As a developer of the Cait Library, you don't want to generate a new wheel file and install the new version every time you added a new function. In this case, we recommend to use inside the folder that contains the setup.py file the

.. code:: console

    $ pip install -e .

pip editable option, that includes changes right away. It is also possible to install directly from the git repository, for this there are many tutorials available, e.g. https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/.

Installation from Git
====================================

The easiest way to install this library is to install it directly from git.
Following [ https://pip.pypa.io/en/latest/reference/pip_install/#git ] we only have to
execute the two commands:

.. code:: console

    $ pip install -U wheel setuptools twine
    $ pip install git+https://git.cryocluster.org/fwagner/cait.git[@<branch|tag|commit|...>]

The library can upgrade by simply adding the ```-U``` or ```--upgrade``` flag to the commands above.