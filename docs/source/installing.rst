.. _installation-page:

************
Installation
************

There are various ways for the installation of cait, which are outlined in the following. Usually we provide a stable release, typically hosted on the GitLab/GitHub master branch and a development version. The development version might still be unstable and undocumented in some new features, does however include all cutting-edge and current implementations.
Our recommendation is therefore to **use the development branch**, combined with active bug-reporting in the GitLab/Hub issue tracker. Due to the still very small user and developer community, the stable release is also to be understood as a beta version.

.. note::
  **Important Note for JupyterHub on computing clusters**

  In the past, many users experienced issues with our interactive plotting tools which are based on plotly and ipywidgets. These problems were due to version mismatches between the plotly/ipywidgets packages and their corresponding JupyterLab extensions (which are automatically installed alongside the packages). 

  To not run into such issues in the first place, we recommend one of the following approaches:

  * Install cait in the base environment of your computing cluster's JupyterLab. 
  * Install it in a virtual environment (because you want or have to) is also possible but you will have to make sure that the same plotly/ipywidgets versions (which are installed as dependencies of cait) are also present in the base environment. A good practice to ensure this is to always pip-upgrade plotly/ipywidgets to the latest version in both environments. Note that you will potentially have to match these versions every time you upgrade cait.

  Lastly, remember to **restart JupyterHub completely** (not just the kernel) for the changes to take effect.

  To learn how to add a virtual environment as a kernel to jupyterlab, refer to `this great reference <https://janakiev.com/blog/jupyter-virtual-envs/>`_.

.. note::
    **Installation in virtual environments**

    We recommend to install cait in a separate environment together with a clean installation of jupyter-lab (for interactive analysis) to avoid version conflicts of dependencies:

    .. code:: console

        $ python3 -m venv venv_cait
        $ source venv_cait/bin/activate
        $ python -m pip install —-upgrade pip
        $ python -m pip install jupyterlab
        $ deactivate
        $ source venv_cait/bin/activate
        $ python -m pip install cait
        $ deactivate
        $ source venv_cait/bin/activate
        $ jupyter-lab

    If you want, you can add the newly created virtual environment as a kernel for jupyter (such that you don't have to do the last two lines above, i.e. always activate the environment):

    .. code:: console

        $ source venv_cait/bin/activate
        $ python -m pip install ipykernel
        $ python -m ipykernel install --name=venv_cait

    You can now choose the kernel in jupyter-lab, VS code, etc.

Installation from PyPI
====================================

Cait is hosted on the Python package index.

.. code:: console

    $ python -m pip install cait

For older or unreleased version, use the installation from Git.

There are some additional dependencies which can be installed together with cait using the following syntax:

.. code:: console

    $ pyton -m pip install cait[<opt_dep1>, <opt_dep2>]

- ``remfiles``: Also install libraries needed to access remote files, most prominently using the 'XRootD' protocol.
- ``nn``: Install neural network dependencies which are not installed by default to keep Cait more light-weight.
- ``clplot``: If you are one of the few people who want to get the 'uniplot' backend for 'cait.versatile' plotting classes, use this optional dependency to unlock command line plotting.
- ``test``: Install 'pytest' to run tests.

Options for Developers
======================

As a developer of the Cait Library, it's best if you clone the repository and make an *editable installation*:

.. code:: console

    $ git clone https://gitlab.cern.ch/cryocluster/cait.git
    $ python -m pip install -e cait/

A full copy/paste for installing Cait from the repository (cleanly and reproducably in a virtual environment) is given in the following. We also directly check out the development branch for the most up-to-date features.

.. code:: console

    $ mkdir CAIT
    $ cd CAIT
    $ git clone https://gitlab.cern.ch/cryocluster/cait.git
    $ python3 -m venv venv_cait
    $ source venv_cait/bin/activate
    $ python -m pip install —-upgrade pip
    $ python -m pip install jupyterlab
    $ deactivate
    $ source venv_cait/bin/activate
    $ python -m pip install -e cait/
    $ deactivate
    $ source venv_cait/bin/activate
    $ python -m pip install ipykernel
    $ python -m ipykernel install --name=venv_cait
    $ deactivate
    $ cd cait
    $ git checkout develop

Deactivating/activating the environment all the time makes sure that the latest changes are recognized by jupyter (and all interactive widgets are properly installed).