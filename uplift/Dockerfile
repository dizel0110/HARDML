FROM --platform=amd64 ubuntu:18.04

LABEL maintainer="antiguru110894@gmail.com" version="0.1.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       git \
       curl \
       wget \ 
       unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm -f ~/miniconda.sh 

ENV PATH /opt/conda/bin:$PATH

RUN activate base \
    && conda install python=3.8.5 -y

RUN activate base \
    && pip install --user --no-cache-dir --compile --no-deps causalml

COPY requirements.txt /root/requirements.txt
RUN pip install --user --no-cache-dir --compile -r ~/requirements.txt \
    && rm -f ~/requirements.txt