#pragma once

#include "matrix.hpp"

namespace mlk {

template<typename T>
void naive_gemm(const Matrix<T>& left,
                const Matrix<T>& right,
                Matrix<T>& output
           );

template<typename T>
void reordered_gemm(const Matrix<T>& left,
                    const Matrix<T>& right,
                    Matrix<T>& output
           );


template<typename T>
void tiled_gemm_a(const Matrix<T>& left,
                const Matrix<T>& right,
                Matrix<T>& output,
                std::size_t max_block_size
           );

template<typename T>
void tiled_gemm_b(const Matrix<T>& left,
                const Matrix<T>& right,
                Matrix<T>& output,
                std::size_t max_block_size
           );

template<typename T>
void multithreaded_gemm_a(const Matrix<T>& left,
                          const Matrix<T>& right,
                          Matrix<T>& output,
                          std::size_t max_block_size,
                          std::size_t num_threads
           );

template<typename T>
void multithreaded_gemm_b(const Matrix<T>& left,
                          const Matrix<T>& right,
                          Matrix<T>& output,
                          std::size_t max_block_size,
                          std::size_t num_threads
           );

template<typename T>
void multithreaded_gemm_c(const Matrix<T>& left,
                          const Matrix<T>& right,
                          Matrix<T>& output,
                          std::size_t num_threads
);


enum class GemmKernels {
    Naive = 0,
    Reordered = 1,
    Tiled_a = 2,
    Tiled_b = 3,
    Multithreaded_a = 4,
    Multithreaded_b = 5,
    Multithreaded_c = 6
};

}

template <typename T>
using GemmFunctionPtr = void (*)(const mlk::Matrix<T>&,
                                 const mlk::Matrix<T>&,
                                 mlk::Matrix<T>&
                                 );
