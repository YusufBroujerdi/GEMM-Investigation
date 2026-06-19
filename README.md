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

## Build instructions

The project requires an anaconda environment to build. The environment.yaml file specifies an environment which can be used to build the project and run any Python script. Otherwise, I build the project using the shell scripts in /scripts:

```bash
conda activate project_environment
(project_environment) ./scripts/release_gcc.sh
```

## Results

An overview of the results can be found in this same folder, in RESULTS.md:

* [Results](./RESULTS.md)

## Reproducing Results

Every chart and table in the RESULTS.md is sourced directly from a python function defined in reproducible_results.py in benchmarking. So, for example, generating the table showing the first comparison of gemms by indeces can be derived by running the following from the project root:

```bash
(project_environment) cd benchmarking
(project_environment) python3
>>> from reproducible_results import compare_index_strategies
>>> compare_index_strategies()
```

More details can be found in:

* [Benchmarking](./benchmarking/README.md)