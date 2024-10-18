FROM ubuntu:22.04

LABEL creator="leehwansoo"
LABEL version="1.0.0"
LABEL description="ubuntu C++ Development environment"

# Work Directory
WORKDIR /usr/src/leehwansoo

RUN apt update && apt upgrade -y

# Package Install
RUN apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    wget \
    cmake \
    clang-format \
    ninja-build \
    git \
    doxygen \
    doxygen-gui \
    graphviz

# Ubuntu Toolchain PPA repository Add
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get upgrade -y

# LLVM Install
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 14

# Commands to run when running a docker container
CMD ["bash"]
