# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    tzdata wget git git-lfs libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev gcc \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    python3-pip libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev gosu \
    libpq-dev python3-dev build-essential && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 將 python 指令指向 python3
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --no-cache-dir -U pip setuptools wheel

COPY . /usr/local/PyFace
RUN cd /usr/local/PyFace && python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && rm -rf /usr/local/PyFace

# Preload data
RUN python -c "import pyface"

WORKDIR /code

CMD ["bash"]