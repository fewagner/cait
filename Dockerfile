FROM python:3.10.13-slim
MAINTAINER SAMIR BANIK

# work in virtual environment
ENV VIRTUAL_ENV=/opt/venv_container
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# copy cait repository
COPY . /cait

# upgrade pip, install jupyterhub/lab and cait (important: cait last for lab widget dependencies!)
RUN python -m pip install --upgrade pip \
    && python -m pip install https://github.com/jupyterhub/batchspawner/archive/main.zip \
    && python -m pip install jupyterhub \
    && python -m pip install jupyterlab \
    && python -m pip install -e /cait

# install dcap library
RUN apt-get update && apt-get install -y dcap-dev

# configure jupyter
# RUN jupyterhub --generate-config

#ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpdcap.so