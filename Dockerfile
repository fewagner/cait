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
RUN apt-get update -qq && apt-get install -y -qq hdf5-tools git nano cmake

# upgrade pip, install jupyterhub/lab and cait (important: cait last for lab widget dependencies!)
RUN python -m pip install --upgrade pip \
    && python -m pip install https://github.com/jupyterhub/batchspawner/archive/main.zip \
    && python -m pip install jupyterhub \
    && python -m pip install jupyterlab==3.4.0 \
    && python -m pip install -e /opt/programs/cait \
    && python -m pip install xrootd