FROM --platform=amd64 ubuntu:18.04

LABEL maintainer="antiguru110894@gmail.com" version="0.2.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       libjpeg-dev \
       libpng-dev \
       git \
       curl \
       wget \ 
       unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm -f ~/miniconda.sh 

ENV PATH /opt/conda/bin:$PATH

COPY requirements.txt /root/requirements.txt
RUN activate base \
    && pip install --user -r ~/requirements.txt 

