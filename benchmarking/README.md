# Benchmarking Workflow

## Benchmarking steps

This folder contains two python files: 

* benchmarking.py offers a helper function evaluate_specs which automates the whole pipeline, in addition to a Spec and Specs class for easy test specification.

* reproducible_results.py has a self-contained python function for each chart/table/collection of charts/tables included in the results markdown. This ensures the exact logic for the tests can be reviewed and reproduced.

Each chart/table/collection of charts/tables follows the following pipeline:

1) A call is made to a function in reproducible_results.py

2) The function generates a collection of specifications wrapped in the Specs class from benchmarking.py. These are passed to evaluate_specs() in benchmarking.py

3) evaluate_specs takes the Specs instance and generates a csv called spec.csv. This csv is passed to the executable bench_gemm, which is built in CMake's release mode.

4) bench_gemm processes the results and generates a results.csv, which is in turn passed back through evaluate_specs to the original function in reproducible_results.py

Generally, I explore different GEMM implementations using a jupyter notebook importing benchmarking.py, and reproducible findings are then added to reproducible_results.py

## System details

These tests were run with an i5-12450H x86 CPU, with a Linux Mint operating system.

My i5-12450H has 8 cores:

1) 4 efficiency cores (1 thread each, 2MiB shared L2 cache)
2) 4 performance cores (2 threads each, 1.25MiB separate L2 caches)

Naturally, specific results will vary between systems.


### [Results Overview](../RESULTS.md)

### [Main README](../README.md)