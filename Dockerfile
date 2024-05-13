From debian:buster-slim
MAINTAINER SAMIR BANIK

RUN apt-get update && apt-get install git libpython3-dev python3-pip

RUN python3 -m pip install numpy

RUN apt-get update && apt-get install dcap-dev

COPY . /cait

RUN cd /cait && pip install -e cait

