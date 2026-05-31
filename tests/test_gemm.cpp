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

constexpr size_t test_tile_width = 16;

template <typename T>
void tiled_gemm_a_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output
    ) {
        mlk::tiled_gemm_a(left, right, output, test_tile_width);
    };

template <typename T>
void tiled_gemm_b_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output
    ) {
        mlk::tiled_gemm_b(left, right, output, test_tile_width);
    };


template <typename T>
mlk::validation test_gemm_on_constants(const mlk::GemmTestCase<T>& testcase,
                           GemmFunctionPtr<T> gemm,
                           std::filesystem::path details_path) {

    std::ofstream file{details_path, std::ios::app};
    
    mlk::Matrix<T> candidate{testcase.left().rows(), testcase.right().cols()};
    gemm(testcase.left(), testcase.right(), candidate);

    if (mlk::max_abs_diff(candidate, testcase.output()) > mlk::tolerance<T>)
        {                
        file << "Test case " << testcase.name() << " has failed.\n";
        testcase.write(file);
        file << "Candidate ";
        candidate.write(file);
        return mlk::validation::Failed;
        }

    file << "test passed successfully\n";
    return mlk::validation::Passed;
}

int main(int argc, char* argv[]) {

    mlk::validation test_state = mlk::validation::Passed;

    if (argc != 2)
        throw std::invalid_argument(
            "Give a single argument representing the path for test details.");

    std::filesystem::path path{argv[1]};
    //Clear the old file if exists
    std::filesystem::remove(path);

    {
    using namespace gemm_sanity_tests;
    for (const auto& test : tests) {

        if (test_gemm_on_constants(test, mlk::naive_gemm<float>, path) == mlk::validation::Failed) {
            std::cout << "Naive GEMM failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
        if (test_gemm_on_constants(test, mlk::reordered_gemm<float>, path) == mlk::validation::Failed) {
            std::cout << "Reordered GEMM failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        } 
        if (test_gemm_on_constants(test, tiled_gemm_a_wrapped<float>, path) == mlk::validation::Failed) {
            std::cout << "Tiled GEMM A failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
        if (test_gemm_on_constants(test, tiled_gemm_b_wrapped<float>, path) == mlk::validation::Failed) {
            std::cout << "Tiled GEMM B failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
    }
    }

    {
    using namespace gemm_python_tests;
    for (const auto& test : tests) {

        if (test_gemm_on_constants(test, mlk::naive_gemm<double>, path) == mlk::validation::Failed) {
            std::cout << "Naive GEMM failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
        if (test_gemm_on_constants(test, mlk::reordered_gemm<double>, path) == mlk::validation::Failed) {
            std::cout << "Reordered GEMM failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
        if (test_gemm_on_constants(test, tiled_gemm_a_wrapped<double>, path) == mlk::validation::Failed) {
            std::cout << "Tiled GEMM A failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
        if (test_gemm_on_constants(test, tiled_gemm_b_wrapped<double>, path) == mlk::validation::Failed) {
            std::cout << "Tiled GEMM B failed on: " << test.name() << "\n";
            test_state = mlk::validation::Failed;
        }
    }
    }
    
    return static_cast<int>(test_state);
}
