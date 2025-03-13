****************
Tips and Tricks
****************

Here we put together some tips for usage and development of Cait, that we found useful. We hope they help you in your work!

Containers
==========

If you work on a server without sudo rights, you might not be able to install packages properly. In this case we often use Singularity containers, which are faster than virtual environments, and run our software inside.

Pre-built Docker Containers
---------------------------
For people who have access to the CERN GitLab, the easiest way to get a container is

.. code:: console

    $ singularity pull --docker-login docker://gitlab-registry.cern.ch/cryocluster/cait:<tag>

where ``tag`` could be ``develop`` or any (tagged) release. For each tag, there is a regular and a ``-slim`` version, where ``-slim`` does not include 'heavy' dependencies like ``torch`` and is therefore smaller in size.
If you do not have access to the CERN GitLab, the docker container can be built with this dockerfile `Dockerfile <https://github.com/fewagner/cait/blob/develop/Dockerfile>`_. Refer to the `Docker Documentation <https://docs.docker.com/build/concepts/dockerfile/>`_ on how to use it. Note that we use singularity to pull and run the container, even though it has been built using Docker. This works and is just a matter of preference.

You can use this container e.g. for cluster jobs (see e.g. :ref:`slurm-example-section`) or you can simply run a python session inside the container

.. code:: console

    $ singularity run cait_develop.sif

Building your own Singularity container
---------------------------------------
You typically create the container on our local linux system and copy it then ("scp ...") to the server. We suggest to copy the container is a compressed file format, e.g. `*.tar`, we experienced problems with scp command and uncompressed containers in the past.

First you need a singularity installation on your machine, you can find instructions on their documentation page: https://sylabs.io/guides/3.0/user-guide/installation.html

Second, you need a container configuration file. One that worked for us is given here, you can put it into a file ´container2010.cfg´.

.. code::

    Bootstrap: docker
    From: ubuntu:20.04

    %post

        export DEBIAN_FRONTEND=noninteractive

        export TZ=Europe/Vienna

        apt-get update

        apt-get install -y git git-lfs python3-pip python-pip-whl zsh screen vim finger openssh-client wget curl libxpm4 python3-tk ffmpeg imagemagick geeqie locales python3-lmdb libxext6 xterm dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev gfortran libssl-dev libpcre3-dev xlibmesa-glu-dev libglew1.5-dev libftgl-dev  libmysqlclient-dev libfftw3-dev libcfitsio-dev graphviz-dev libavahi-compat-libdnssd-dev  libldap2-dev python2-dev libxml2-dev libkrb5-dev libgsl0-dev qt5-default libgfortran4 mmv libtinfo5 htop python3-pyx texlive-science texlive-latex-base texlive-latex-extra texlive-latex-recommended rsync sudo firefox libssl1.1 mupdf evince python3-scipy python3-numpy python3-tables python3-colorama tcl tclsh psmisc graphviz dot2tex locate openafs-client krb5-user kinit openafs-krb5 dvipng bc texlive-fonts-extra texlive-pictures iputils-ping autossh tmux tcllib nmap mtr gnuplot python3-gnuplotlib libreoffice-java-common unoconv default-jre gcc-7 gnuplot-x11 aptitude libxft-dev flex bison eog cm-super-minimal python-is-python2 fgallery g++-9 hdfview hdf5-tools

        pip3 install pyyaml pyslha unum scipy numpy==1.20 torch torchvision sympy matplotlib pip jupyter h5py tables plotly pandas ipython cython colorama pyexcel_ods ordered_set reportlab pypdf2 pygraphviz pympler pyfeyn pyhf typing sklearn sphinx_rtd_theme requests datetime bibtexparser jaxlib jax coverage progressbar setuptools>=47.1.1 wheel twine pickle-mixin numba uproot awkward1 pytorch-lightning tqdm ipykernel jupyter_contrib_nbextensions pandas plotly dash jupyter_dash jupyterlab jupyter-server-proxy ipywidgets

        locale-gen "en_US.UTF-8"

        locale-gen "en_US"

        dpkg-reconfigure --frontend noninteractive tzdata

        jupyter contrib nbextension install

        pip3 install cait

    %help

        This is an example for an ubuntu container to run Cait.

Last, you need to type correct commands for the container creation. We put them together in a script, which is below.
Be aware of the sandbox flag, that makes installation of additional packages inside the container possible, but might
lead to issues with copying the container.

.. code:: bash

    #!/bin/sh

    CONTAINER=ubuntu2010.simg
    rm -rf $CONTAINER
    sudo singularity build --sandbox $CONTAINER container2010.cfg

You can then always start the container by typing the following command in the same directory. We like to keep it in a script as well.

.. code:: console

    $ singularity shell -c -B /home/,/mnt/,/remote/ -s /bin/bash -H /home/USERNAME/ --writable ./ubuntu2010.simg

Please look into the Singularity manual for details of above command. You might have to adapt several paths, according to
your system. E.g. with the -B flag, you can bin directories from the server within the the container, you will need this for accessing data.
In this example, the data we want to access is in /mnt/ and /remote/. However, sometimes the binding is incompatible with the --writeable
flag, which makes the installation of additional packages possible. In this case, you need to start either with --writeable or with the mounted folder.

.. _slurm-example-section:

SLURM job example
=================
You will not do large scale analysis in jupyter. Rather, you would write a small script, let's call it ``analyse_one_file.py``, which performs the analysis on one file. It takes three command line arguments: the file name, the file directory, and an output directory, then it performs some analysis:

.. code:: python

    import sys 

    f_name, f_dir, out_dir = sys.argv[1:]
    # do something with file

Then you have a text file ``filelist.txt`` with a list of files to process:

.. code::

    file1.bin
    file2.bin
    file3.bin

Finally, you define a script ``analysis_job.sh`` which is used to submit a job for each file to the cluster: 

.. code:: bash

    #!/bin/bash

    # here, we use the cait develop container described above
    CONTAINER_NAME="cait_develop"
    # path of the container
    SINGULARITY_PATH="/path/to/container"

    SIF_CONTAINER=${SINGULARITY_PATH}/${CONTAINER_NAME}".sif"

    # mail notifications
    SLURM_SBATCH_MAIL=" --mail-type=ALL"
    SLURM_SBATCH_MAIL+=" --mail-user=my.email@address.at"

    # bind directories to container (otherwise files cannot be accessed)
    BIND_STORAGE="${HOME}"
    
    # script name and directories
    SCRIPT=${HOME}/analyse_one_file.py
    DATA_IN_DIR=${HOME}/path/to/input/files
    DATA_OUT_DIR=${HOME}/path/where/results/should/be/saved

    # directory where log/error files are written
    JOB_DIR=${HOME}/analysis_job_files

    # the text file with the files to process
    FILE_LIST=$1

    # --------------------------------------
    for file in $(cat $FILE_LIST)
    do
        SLURM_OPTIONS_SBATCH=" "
        SLURM_OPTIONS_SBATCH+=" --job-name=cait-analysis"
        SLURM_OPTIONS_SBATCH+=" --chdir=${JOB_DIR}"
        SLURM_OPTIONS_SBATCH+=" --output=log_${file}.out"
        SLURM_OPTIONS_SBATCH+=" --error=log_${file}.err"
        SLURM_OPTIONS_SBATCH+=" --partition=c"
        SLURM_OPTIONS_SBATCH+=" --time=6:00:00"
        SLURM_OPTIONS_SBATCH+=" --qos=c_medium"
        SLURM_OPTIONS_SBATCH+=" --parsable"
        SLURM_OPTIONS_SBATCH+=" --cpus-per-task=8"
        #SLURM_OPTIONS_SBATCH+=" --kill-on-invalid-dep=yes"
        SLURM_OPTIONS_SBATCH+=" --mem=16G"   

        # depending on your script, you have to change the following
        ARGS_SCRIPT="${file} ${DATA_IN_DIR} ${DATA_OUT_DIR}" 

        # put everything together for the cluster to process
        RUN_COMMAND="singularity run ${SIF_CONTAINER} python ${SCRIPT} ${ARGS_SCRIPT}"
        SBATCH_WRAP=${SBATCH_WRAP_OPTIONS}" srun "${RUN_COMMAND}
        
        printf "\n\n-----------------------------\n SLURM batch jobs: run command \n"
        echo " sbatch ${SLURM_OPTIONS_SBATCH} --wrap=\"${SBATCH_WRAP}\""
        
        jobID_analysis=$(sbatch ${SLURM_OPTIONS_SBATCH} --wrap="${SBATCH_WRAP}")

    done

    printf "\n\nExit script $0 \n"

    exit 0

To submit the job, you run 

.. code:: console

    $ source analysis_job.sh filelist.txt

To see running jobs, use ``squeue -u <your>.<username>``. To attach to a running job, use ``sattach <jobID>.0`` (you can exit after attaching using ``CTRL+C``).

Notebooks on a Server
=====================

Large scale data processing is typically not done locally but on a remote server. In case we have no X forwarding available
for the remote server, we can still use Jupyter Notebooks for easily accessible visualizations. A very simple, 3-step description
how to run a notebook on a server, but get the output in you local browser, can be found on this homepage:
https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/

In case you run on a server with SLURM (e.g. the CLIP in Vienna), here is a tutorial for how to start the jupyter notebook
with SLURM:
https://alexanderlabwhoi.github.io/post/2019-03-08_jpn-slurm/

Virtual Terminal
=========================

In case you are working on a server and experience troubles due to an instable internet connection, or need to run scripts and shutdown
your machine while they are running, you can use a virtual terminal multiplexer. We like to use `screen` (https://linuxize.com/post/how-to-use-linux-screen/).
You can start screen on the server, before executing your scripty or starting up your Jupyter kernel. The screen session keeps running,
even if you disconnect the ssh connection to the server. At any later point, you can reattach to the screen session and continue working or watch outputs of your scripts.

Contents of HDF5 Files
=========================

There are several tools to view the contents of HDF5 files. For local work or if X-forwarding is available, we recommend
HDFView and VITables. If the contents must be listed directly in the command line, we recommend h5dump and h5ls.

Remote Visualization
=========================

Many server clusters provide a remote visualization service for Jupyter Notebooks, eg. the MPCDF (https://rvs.mpcdf.mpg.de/)
and the CLIP (https://jupyterhub.vbc.ac.at/hub/home, VPN needed). We like to use these services for all interactive work (creation of SEV, Filter, ...) and scripts for long-lasting jobs (triggering, fit and feature pipelines, ...).

Debugging
=============

A usefull tool for  debugging code is the library **IPython pdb** (https://pypi.org/project/ipdb/).
This library exports functions to access the IPython debugger, which features tab completion, syntax highlighting, better tracebacks, better introspection with the same interface as the pdb module.

.. code:: console

    $ pip install ipdb

Adding the line

.. code:: python

    import ipdb; ipdb.set_trace()

any where in your code halts the execution and lets insert and execute additional lines.
