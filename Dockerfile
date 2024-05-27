FROM rayproject/ray:latest

ENV TZ=Europe/Prague \
    DEBIAN_FRONTEND=noninteractive 

RUN sudo apt-get update
RUN sudo apt-get install -y build-essential
RUN sudo apt-get install -y git
RUN sudo apt-get install -y lsb-release
RUN sudo apt-get install -y libspdlog-dev
RUN sudo apt-get install -y libeigen3-dev
RUN sudo apt-get install -y software-properties-common
RUN sudo apt-get install -y wget    
RUN sudo apt-get install -y vim


RUN sudo apt purge --auto-remove cmake

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' 
RUN sudo apt-get update
RUN sudo apt-get install -y cmake

WORKDIR /work

RUN git clone https://github.com/google/or-tools.git -b stable
RUN cd or-tools && cmake -S. -Bbuild -DBUILD_DEPS:BOOL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
RUN cd or-tools && sudo cmake --build build --target install -j 10

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
ENV PATH=/root/miniconda3/bin:$PATH
RUN conda init bash

RUN conda install -c conda-forge gcc=12.2.0
RUN conda install -n base pybind11
RUN conda install -n base -c rapidsai-nightly cmake_setuptools
RUN conda install -n base -c conda-forge gymnasium
RUN conda install -n base -c conda-forge hydra-core
RUN conda install -n base -c anaconda pandas
RUN conda install -n base -c anaconda numpy
RUN conda install -n base -c anaconda matplotlib

COPY docker_build/rats /work/rats
RUN sudo chmod -R 777 /work/rats
RUN conda run -n base sh -c "export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && cd /work/rats && pip install ."
