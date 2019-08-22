FROM ubuntu:16.04
MAINTAINER "Jorge Espinoza" <jorge@jabud.tech>

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev libpq-dev\
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

ADD . /prod

RUN cd /prod && pip3 install -r requirements.txt

WORKDIR /prod