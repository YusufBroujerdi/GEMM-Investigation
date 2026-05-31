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

struct TileState;

template<typename T>
void tiled_gemm_b(const Matrix<T>& left,
                const Matrix<T>& right,
                Matrix<T>& output,
                std::size_t max_block_size
           );

template<typename T>
void tiled_gemm_b(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    TileState& ts
);

}

template <typename T>
using GemmFunctionPtr = void (*)(const mlk::Matrix<T>&,
                                 const mlk::Matrix<T>&,
                                 mlk::Matrix<T>&
                                 );
