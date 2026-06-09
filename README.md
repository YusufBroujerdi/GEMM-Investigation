# GEMM Performance Engineering Project

By Yusuf Broujerdi

## Overview

The goal of this project is to understand the performance trade-offs of different GEMM implementations, in order to gain a deeper understanding of CPU architecture.

This project also implements a full workflow for testing and benchmarking different GEMMS, to enable fast iteration. Python scripts drive the benchmarking and generate unit tests, while the core implementations live entirely in C++.

## Further detail

Links can be found for more detail on the following:

* [Benchmarking](./benchmarking/README.md)
* [Testing](./tests/README.md)
* [C++ Source](./src/README.md)
* [GEMM implementations](./src/gemms/README.md)

## Results

An overview of the results can be found in this same folder, in RESULTS.md:

* [Results](./RESULTS.md)