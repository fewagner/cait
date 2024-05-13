FROM python:3.10.13-slim
MAINTAINER SAMIR BANIK

#update pip
RUN python -m pip install --upgrade pip

#install numpy (needed for venv install) and python-magic (helpful)
RUN pip install numpy && pip install python-magic  && pip install ipympl


RUN apt-get update && apt-get install -y dcap-dev

COPY . /cait

RUN cd /cait && pip install -e ../

RUN pip install https://github.com/jupyterhub/batchspawner/archive/main.zip && pip install jupyterhub

