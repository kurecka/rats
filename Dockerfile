FROM rayproject/ray:latest

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
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' 
RUN apt-get update
RUN apt-get install -y cmake

WORKDIR /work

RUN git clone https://github.com/google/or-tools.git -b stable
RUN cd or-tools && cmake -S. -Bbuild -DBUILD_DEPS:BOOL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
RUN cd or-tools && cmake --build build --target install -j 10

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
RUN chmod -R 777 /work/rats
RUN conda run -n base sh -c "export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && cd /work/rats && pip install ."

# RUN wget https://github.com/prometheus/prometheus/releases/download/v2.47.0/prometheus-2.47.0.linux-amd64.tar.gz -O /work/prometheus.tar.gz && tar -xzf /work/prometheus.tar.gz && rm /work/prometheus.tar.gz
# run in background
# RUN cd prometheus-2.47.0.linux-amd64

# RUN apt-get install -y adduser libfontconfig1 musl
# RUN wget https://dl.grafana.com/enterprise/release/grafana-enterprise_10.1.2_amd64.deb
# RUN dpkg -i grafana-enterprise_10.1.2_amd64.deb && rm grafana-enterprise_10.1.2_amd64.deb
# RUN apt-get install -y gnupg2 curl
# RUN curl https://packages.grafana.com/gpg.key | apt-key add -
# RUN add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
# RUN apt-get update
# RUN apt-get -y install grafana
