****
cait
****

Cait (Cryogenic Artificial Intelligence Tools) is a Python 3 software package for the machine learning based analysis
of raw data from cryogenic dark matter experiments. It is tailored to the needs of the CRESST and COSINUS experiment,
but applicable to other, similar data structures.

**Documentation:** Preview of new version: https://adoring-hoover-f7972f.netlify.app/

**Source Code:** https://git.cryocluster.org/fwagner/cait (for CRYOCLUSTER collaborations), sync on: https://github.com/fewagner/cait

**Bug Report:** https://git.cryocluster.org/fwagner/cait/-/issues (for CRYOCLUSTER collaborations), otherwise: https://github.com/fewagner/cait/issues


Installation
============

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

Version History
===============

Master branch is on Version 1.0.0 and stable.

Version numbers follow the segmantic versioning guide (https://semver.org/).

We want you ...
===============

... to contribute! We are always happy about any contributions to our software. To coordinate
efforts, please get in touch with felix.wagner(at)oeaw.ac.at such that we can include your
features in the upcoming release. If you have any troubles with the current release, please open an issue in the Bug Report.