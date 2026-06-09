# Benchmarking Workflow

The executable bench_gemm in /apps/ takes a csv of benchmark specifications, and outputs another csv of results. The relative locations of these csvs can be controlled via shell commands to the executable.

This folder contains two python files: 

* benchmarking.py offers a helper function evaluate_specs which automates the whole pipeline, in addition to a Spec and Specs class for easy test specification.

* reproducible_results.py has a self-contained python function for each chart/table/collection of charts/tables included in the results markdown. This ensures the exact logic for the tests can be reviewed and reproduced.

Generally, I explore different GEMM implementations using a jupyter notebook importing benchmarking.py, and reproducible findings are then added to reproducible_results.py


### [Results Overview](../RESULTS.md)

### [Main README](../README.md)