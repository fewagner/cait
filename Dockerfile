FROM python:3.10.13-slim
MAINTAINER SAMIR BANIK

#update pip
RUN python -m pip install --upgrade pip

#install numpy (needed for venv install) and python-magic (helpful)
RUN pip install numpy && pip install python-magic  && pip install ipympl


RUN apt-get update && apt-get install -y dcap-dev

COPY . /cait

RUN pip install -e /cait

RUN pip install https://github.com/jupyterhub/batchspawner/archive/main.zip && pip install jupyterhub

ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpdcap.so