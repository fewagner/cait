FROM python:3.10.13-slim

# work in virtual environment
ENV VIRTUAL_ENV=/opt/venv_container
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ensure jupyter searches virtual env first (see https://discourse.jupyter.org/t/jupyter-paths-priority-order/7771)
# (see also https://docs.jupyter.org/en/latest/use/jupyter-directories.html)
ENV JUPYTER_PREFER_ENV_PATH=1

# copy cait repository
COPY . /opt/programs/cait

# install hdf5 tools, git and nano
RUN apt-get update -qq && apt-get install -y -qq hdf5-tools git nano cmake davix-dev g++ libcurl4-openssl-dev libfuse-dev \
                                                 libgtest-dev libisal-dev libjson-c-dev libkrb5-dev libmacaroons-dev libreadline-dev \
                                                 libscitokens-dev libssl-dev libsystemd-dev libtinyxml-dev libxml2-dev make \
                                                 pkg-config python3-dev python3-setuptools uuid-dev voms-dev zlib1g-dev

# upgrade pip, install jupyterhub/lab and cait (important: cait last for lab widget dependencies!). Fix jupyterlab version because in 4.2.0, the code cells are weird
RUN python -m pip install --upgrade pip \
    && python -m pip install https://github.com/jupyterhub/batchspawner/archive/main.zip \
    && python -m pip install jupyterhub \
    && python -m pip install jupyterlab==3.4.0 \
    && python -m pip install -e /opt/programs/cait[nn,clplot,remfiles]