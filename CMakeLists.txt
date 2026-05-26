cmake_minimum_required(VERSION 3.10)

project(cpp_ml_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")

set(GENERATED_TEST_CASES
    ${CMAKE_CURRENT_BINARY_DIR}/generated_test_cases
)

add_custom_command(
    OUTPUT
        ${GENERATED_TEST_CASES}/gemm_python_tests.cpp
        ${GENERATED_TEST_CASES}/include/gemm_python_tests.hpp
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/generate_test_cases.py
        ${GENERATED_TEST_CASES}/gemm_python_tests.cpp
        ${GENERATED_TEST_CASES}/include/gemm_python_tests.hpp
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/generate_test_cases.py
    COMMENT
        "Generating Python test cases"
)

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
    ${GENERATED_TEST_CASES}/gemm_python_tests.cpp
)

target_include_directories(testcases
    PUBLIC
        tests/test_cases/include
        ${GENERATED_TEST_CASES}/include
)

target_link_libraries(testcases
    PUBLIC
        mlkernels
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

add_test(
    NAME
        test_gemm
    COMMAND
        test_gemm ${GENERATED_TEST_CASES}/test_details.txt
)

