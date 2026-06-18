# Testing

Every GEMM implementation is tested on several hardcoded C++ sanity tests and an extensible suite of python tests.

Whenever the project is built, the python script generate_test_cases.py generates a range of matrix definitions which are in turn compiled and tested. These include a range of shapes and sizes with results validated against the result of matrix multiplication in numpy.

The sanity tests are done with floats, whereas the python tests with doubles. This difference allows both types to be tested, while the larger python matrices receive more precision.

In addition to all of the above, every benchmark test outputs validation results when comparing against gemm_reordered (a straightforward reference implementation). This is an opportunity to confirm validity on larger matrices than the unit tests offer.

### [Main README](../README.md)