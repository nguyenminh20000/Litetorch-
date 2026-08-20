FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY include/ /app/include/
COPY src/ /app/src/
COPY tests/ /app/tests/

RUN g++ -std=c++14 -O3 -Iinclude src/*.cpp tests/demo_run.cpp -o litetorch_demo -lpthread -ldl

ENTRYPOINT ["./litetorch_demo"]
