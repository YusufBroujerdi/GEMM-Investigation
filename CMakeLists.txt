cmake_minimum_required(VERSION 3.10)

project(cpp_ml_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(mlkernels
    src/gemm_naive.cpp
    src/gemm_reordered.cpp
    src/validate.cpp
)

target_include_directories(mlkernels
    PUBLIC
        include
)

add_library(testcases
    tests/test_cases/gemm_sanity_tests.cpp
)

target_include_directories(testcases
    PUBLIC
        include
        tests/test_cases/include
)

add_executable(bench_gemm
    apps/bench_gemm.cpp
)

target_link_libraries(bench_gemm
    PRIVATE
        mlkernels
)

add_executable(test_gemm
    tests/test_gemm.cpp
)

target_link_libraries(test_gemm
    PRIVATE
        mlkernels
        testcases
)

enable_testing()

add_test(NAME test_gemm COMMAND test_gemm)

