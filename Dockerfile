FROM ubuntu:jammy


ENV TZ=Europe/Prague \
DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y lsb-release
RUN apt-get install -y libspdlog-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y software-properties-common
RUN apt-get install -y wget    
RUN apt-get install -y vim

RUN apt purge --auto-remove cmake

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ jammy main"
RUN apt-get update
RUN apt-get install -y cmake


ARG num_jobs=16

WORKDIR /build
RUN git clone https://github.com/google/or-tools.git -b stable
WORKDIR /build/or-tools
RUN cmake -S. -Bbuild -DBUILD_DEPS:BOOL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
RUN cmake --build build --target install -j $num_jobs

# RUN mkdir -p ~/miniconda3
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# RUN rm -rf ~/miniconda3/miniconda.sh
# ENV PATH=/root/miniconda3/bin:$PATH
# RUN conda init bash

# RUN conda install -c conda-forge gcc=12.2.0
# RUN conda install -n base pybind11
# RUN conda install -n base -c rapidsai-nightly cmake_setuptools
# RUN conda install -n base -c conda-forge gymnasium
# RUN conda install -n base -c conda-forge hydra-core
# RUN conda install -n base -c anaconda pandas
# RUN conda install -n base -c anaconda numpy
# RUN conda install -n base -c anaconda matplotlib
RUN apt-get install -y python3-pip

RUN pip install "pybind11[global]"
RUN pip install gymnasium
RUN pip install cmake_setuptools
RUN pip install pandas
RUN pip install numpy
RUN pip install --upgrade build

RUN apt install -y python3-venv

WORKDIR /work/rats
COPY . .
RUN pip install -e .
