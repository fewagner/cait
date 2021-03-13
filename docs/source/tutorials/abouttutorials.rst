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

Environment configuration
=========================

As Cait is right now not tested on many different systems, but depends on several underlying packages, we include in the
following the output of 'pip freeze' on the system where we developed large parts of Cait. This includes the version
numbers of all dependencies, in case we forgot to include a package in the requirements properly. If this is the case,
please open an issue in the Cait GitLab repository.

.. code:: bash

    $ (venv) felix@Felixs-MacBook-Pro cait % pip freeze
    absl-py==0.11.0
    alabaster==0.7.12
    appdirs==1.4.4
    appnope==0.1.0
    argon2-cffi==20.1.0
    asteroid-sphinx-theme==0.0.3
    asttokens==2.0.4
    async-generator==1.10
    atomicwrites==1.4.0
    attrs==20.3.0
    awkward==0.14.0
    awkward1==0.4.3
    Babel==2.9.0
    backcall==0.2.0
    bleach==3.2.1
    cachetools==4.1.1
    -e git+git@git.cryocluster.org:fwagner/cait.git@3b3b08063d82e025702302e53c21a64166074ab4#egg=cait
    certifi==2020.11.8
    cffi==1.14.4
    chardet==3.0.4
    colorama==0.4.4
    commonmark==0.9.1
    csaps==1.0.3
    cycler==0.10.0
    dataclasses==0.6
    decorator==4.4.2
    defusedxml==0.6.0
    docutils==0.16
    entrypoints==0.3
    executing==0.5.4
    fsspec==0.8.4
    future==0.18.2
    gitdb==4.0.5
    GitPython==3.1.13
    google-auth==1.23.0
    google-auth-oauthlib==0.4.2
    gplearn==0.4.1
    grpcio==1.33.2
    gspread==3.6.0
    h5py==3.1.0
    httplib2==0.18.1
    icecream==2.0.0
    idna==2.10
    imageio==2.9.0
    imagesize==1.2.0
    importlib-metadata==3.4.0
    ipykernel==5.4.2
    ipython==7.19.0
    ipython-genutils==0.2.0
    ipywidgets==7.5.1
    jedi==0.17.2
    Jinja2==2.11.2
    joblib==0.17.0
    jsonschema==3.2.0
    jupyter==1.0.0
    jupyter-cache==0.4.2
    jupyter-client==6.1.7
    jupyter-console==6.2.0
    jupyter-core==4.7.0
    jupyter-sphinx==0.3.1
    jupyterlab-pygments==0.1.2
    keyring==21.5.0
    kiwisolver==1.3.1
    livereload==2.6.3
    llvmlite==0.35.0
    Markdown==3.3.3
    markdown-it-py==0.6.2
    MarkupSafe==1.1.1
    matplotlib==3.3.3
    mdit-py-plugins==0.2.5
    mistune==0.8.4
    more-itertools==8.6.0
    myst-nb==0.11.1
    myst-parser==0.13.5
    nbclient==0.5.1
    nbconvert==5.6.1
    nbdime==2.1.0
    nbformat==5.0.8
    nest-asyncio==1.4.3
    networkx==2.5
    notebook==6.1.5
    numba==0.52.0
    numpy==1.19.4
    oauth2client==4.1.3
    oauthlib==3.1.0
    packaging==20.4
    pandas==1.1.4
    pandocfilters==1.4.3
    parso==0.7.1
    patsy==0.5.1
    pexpect==4.8.0
    pickle-mixin==1.0.2
    pickleshare==0.7.5
    Pillow==8.0.1
    pkginfo==1.6.1
    plotly==4.13.0
    pluggy==0.13.1
    prometheus-client==0.9.0
    prompt-toolkit==3.0.8
    protobuf==3.14.0
    ptyprocess==0.6.0
    py==1.9.0
    pyasn1==0.4.8
    pyasn1-modules==0.2.8
    pycparser==2.20
    Pygments==2.7.2
    pykalman==0.9.5
    pyparsing==2.4.7
    pyrsistent==0.17.3
    pytest==4.4.1
    pytest-runner==5.2
    python-dateutil==2.8.1
    pytorch-lightning==1.0.6
    pytz==2020.4
    PyWavelets==1.1.1
    PyYAML==5.3.1
    pyzmq==20.0.0
    qtconsole==5.0.2
    QtPy==1.9.0
    readme-renderer==28.0
    recommonmark==0.7.1
    requests==2.25.0
    requests-oauthlib==1.3.0
    requests-toolbelt==0.9.1
    retrying==1.3.3
    rfc3986==1.4.0
    rinoh-typeface-dejavuserif==0.1.3
    rinoh-typeface-texgyrecursor==0.1.1
    rinoh-typeface-texgyreheros==0.1.1
    rinoh-typeface-texgyrepagella==0.1.1
    rinohtype==0.5.0
    rsa==4.6
    scikit-image==0.18.1
    scikit-learn==0.24.1
    scipy==1.5.4
    Send2Trash==1.5.0
    six==1.15.0
    sklearn==0.0
    smmap==3.0.5
    snowballstemmer==2.0.0
    Sphinx==3.3.1
    sphinx-autobuild==2020.9.1
    sphinx-reload==0.2.0
    sphinx-rtd-theme==0.5.1
    sphinx-togglebutton==0.2.3
    sphinxcontrib-applehelp==1.0.2
    sphinxcontrib-devhelp==1.0.2
    sphinxcontrib-htmlhelp==1.0.3
    sphinxcontrib-jsmath==1.0.1
    sphinxcontrib-qthelp==1.0.3
    sphinxcontrib-serializinghtml==1.1.4
    SQLAlchemy==1.3.23
    statsmodels==0.12.1
    tensorboard==2.4.0
    tensorboard-plugin-wit==1.7.0
    terminado==0.9.1
    testpath==0.4.4
    threadpoolctl==2.1.0
    tifffile==2020.12.8
    torch==1.7.0
    torchvision==0.8.1
    tornado==6.1
    tqdm==4.51.0
    traitlets==5.0.5
    tsfel==0.1.3
    twine==3.2.0
    typing-extensions==3.7.4.3
    uproot==3.13.0
    uproot-methods==0.8.0
    uproot4==0.1.2
    urllib3==1.26.2
    wcwidth==0.2.5
    webencodings==0.5.1
    Werkzeug==1.0.1
    widgetsnbextension==3.5.1
    yapf==0.30.0
    zipp==3.4.0