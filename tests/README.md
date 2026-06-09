# Testing

Every GEMM implementation is tested on several hardcoded C++ sanity tests and an extensible suite of python tests.

Whenever the project is built, the python script generate_test_cases.py generates a range of matrix definitions which are in turn compiled and tested.

The sanity tests are done with floats, whereas the python tests with doubles. This difference allows both types to be tested, while the longer python matrices receive more precision.

### [Main README](../README.md)