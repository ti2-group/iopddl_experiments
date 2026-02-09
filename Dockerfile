FROM ubuntu:24.04
ENV CXX=/usr/bin/g++-11
RUN apt update && apt-get install -y cmake g++-11
COPY solver/ /solver
WORKDIR /solver
RUN mkdir build
WORKDIR /solver/build
RUN cmake ..
RUN make -j8

ENTRYPOINT [ "/solver/build/iopddl" ]

