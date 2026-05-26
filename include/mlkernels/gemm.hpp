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

}

template <typename T>
using GemmFunctionPtr = void (*)(const mlk::Matrix<T>&,
                                 const mlk::Matrix<T>&,
                                 mlk::Matrix<T>&
                                 );
