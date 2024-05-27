FROM python:3.10.13-slim
MAINTAINER SAMIR BANIK

# work in virtual environment
ENV VIRTUAL_ENV=/opt/venv_container
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ensure jupyter searches virtual env first (see https://discourse.jupyter.org/t/jupyter-paths-priority-order/7771)
# (see also https://docs.jupyter.org/en/latest/use/jupyter-directories.html)
ENV JUPYTER_PREFER_ENV_PATH=1

# copy cait repository
COPY . /opt/programs/cait

# copy/pasted from CAT container (hopefully this contains everything needed by dcap)
RUN apt-get update && apt-get install -y build-essential curl wget git libpython3-dev libpython-dev nano \ 
cmake libx11-dev libxpm-dev libxft-dev libxext-dev \
libtiff5-dev libgif-dev libgsl-dev libpython-dev libkrb5-dev libxml2-dev libssl-dev \
default-libmysqlclient-dev libpq-dev libqt4-opengl-dev libgl2ps-dev libpcre-ocaml-dev \ 
libgraphviz-dev libdpm-dev unixodbc-dev libsqlite3-dev libfftw3-dev libcfitsio-dev \
dcap-dev libldap2-dev libavahi-compat-libdnssd-dev

# RUN apt-get update && apt-get install -y dcap-dev gcc pkg-config libhdf5-serial-dev

# upgrade pip, install jupyterhub/lab and cait (important: cait last for lab widget dependencies!)
RUN python -m pip install --upgrade pip \
    && python -m pip install https://github.com/jupyterhub/batchspawner/archive/main.zip \
    && python -m pip install jupyterhub \
    && python -m pip install jupyterlab \
    && python -m pip install -e /opt/programs/cait

# copy configuration file for jupyterhub
# COPY .jupyterhub_config.py /opt/venv_container/etc/jupyter/jupyterhub_config.py

# set entrypoint such that the jupyterhub config file is appended to the 'singularity run' command called by CLIP
# ENTRYPOINT ["/usr/bin/bash", "-c", "$@", "--"]