FROM python:3.10.13-slim
MAINTAINER SAMIR BANIK

#update pip
RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y dcap-dev

COPY . /cait

RUN python -m pip install -e /cait

RUN python -m pip install https://github.com/jupyterhub/batchspawner/archive/main.zip
RUN python -m pip install jupyterhub
RUN python -m pip install jupyterlab

#ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpdcap.so