#include <cstring>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "mlkernels/gemm.hpp"
#include "gemm_sanity_tests.hpp"
#include "gemm_python_tests.hpp"


template <typename T>
int test_gemm_on_constants(const mlk::GemmTestCase<T>& testcase,
                           GemmFunctionPtr<T> gemm, std::filesystem::path path) {

    std::size_t i;
    std::ofstream file{path};
    
    mlk::Matrix<T> candidate{testcase.left().rows(), testcase.right().cols()};
    gemm(testcase.left(), testcase.right(), candidate);

    for (i = 0; i < candidate.size(); i++) 
        if (std::abs(candidate.data()[i] - testcase.output().data()[i]) > 0.001)
            {                
            file << "Test case " << i << " has failed.\n";
            testcase.write(file);
            candidate.write(file);
            return 1;
            }

    file << "All tests passed successfully.\n";
    return 0;
}

int main(int argc, char* argv[]) {

    if (argc != 2)
        throw std::invalid_argument("Give a single argument representing path.");

    std::filesystem::path path{argv[1]};

    {
    using namespace gemm_sanity_tests;
    for (const auto& test : tests) {

        if (test_gemm_on_constants(test, mlk::naive_gemm<float>, path) != 0) {
            std::cout << "Naive GEMM failed on: " << test.name();
            return 1;
        }
        if (test_gemm_on_constants(test, mlk::reordered_gemm<float>, path) != 0) {
            std::cout << "Reordered GEMM failed on: " << test.name();
            return 1;
        } 
    }
    }

    {
    using namespace gemm_python_tests;
    for (const auto& test : tests) {

        if (test_gemm_on_constants(test, mlk::naive_gemm<double>, path) != 0) {
            std::cout << "Naive GEMM failed on: " << test.name();
            return 1;
        }
        if (test_gemm_on_constants(test, mlk::reordered_gemm<double>, path) != 0) {
            std::cout << "Reordered GEMM failed on: " << test.name();
            return 1;
        }
    }
    }
    
    return 0;
}
