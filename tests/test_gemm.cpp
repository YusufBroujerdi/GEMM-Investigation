#include <cstring>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "mlkernels/gemm.hpp"
#include "mlkernels/validate.hpp"
#include "gemm_sanity_tests.hpp"
#include "gemm_python_tests.hpp"

template <typename T>
int test_gemm_on_constants(const mlk::GemmTestCase<T>& testcase,
                           GemmFunctionPtr<T> gemm,
                           std::filesystem::path details_path) {

    std::ofstream file{details_path};
    
    mlk::Matrix<T> candidate{testcase.left().rows(), testcase.right().cols()};
    gemm(testcase.left(), testcase.right(), candidate);

    if (mlk::max_abs_diff(candidate, testcase.output()) > mlk::tolerance<T>)
        {                
        file << "Test case " << testcase.name() << " has failed.\n";
        testcase.write(file);
        file << "Candidate ";
        candidate.write(file);
        return 1;
        }

    file << "All tests passed successfully.\n";
    return 0;
}

int main(int argc, char* argv[]) {

    if (argc != 2)
        throw std::invalid_argument(
            "Give a single argument representing the path for test details.");

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
