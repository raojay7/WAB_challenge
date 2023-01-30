FROM  nvidia/cuda:10.2-cudnn7-devel

ADD requirements.txt /

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libsm6 libxext6 libgl1-mesa-glx libxrender-dev ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool sudo \
	cmake protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install 'git+https://gitee.com/wsyin/cocoapi.git#subdirectory=PythonAPI'
# Set a fixed  model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /

