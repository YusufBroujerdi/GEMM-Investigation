# Source Code

This README describes the layout of the source code.

## GEMMS

All of the gemm implementations live in the gemm folder. A brief description of each can be found here:

### [GEMM Descriptions](gemms/README.md)

## matrix.cpp

Defines the templated Matrix class used by GEMM functions and the GemmTestCase class used in testing and benchmarking respectively.

## benchmark.cpp

The benchmark file contains the classes for reading and writing to csvs, as well as the functions which execute the benchmarking, based on a particular specification.

## validate.cpp

The validate file contains various functions for evaluating the closeness of matrices. Useful for validating implementations and measuring any potential precision loss.

## Templating

The Matrix, TestCase and GEMM implementations are templated for floats and doubles. The plan is to see how benchmarks change at different precisions. This is the only use of templating.

### [Main README](../README.md)


# To-Do:

I would like to refactor the code in two major ways:

* Create a MetaGemm class which can hold all the metadata needed for csv reading/writing, benchmarking and testing. So that new GEMMs can be added by only modifying one part of the code, and a single array can hold all the metadata.

* Create a SpecItem class, so that new parameters can be easily added to the spec by instantiating this object, and the GemmBenchmark class can be replaced with a vector of these SpecItems.