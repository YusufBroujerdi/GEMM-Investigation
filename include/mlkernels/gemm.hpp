#pragma once

#include "matrix.hpp"

namespace mlk {

template<typename T>
int naive_gemm(const Matrix<T>& left,
               const Matrix<T>& right,
               Matrix<T>& output
           );

template<typename T>
int reordered_gemm(const Matrix<T>& left,
                   const Matrix<T>& right,
                   Matrix<T>& output
           );

}

template <typename T>
using GemmFunctionPtr = int (*)(const mlk::Matrix<T>&,
                                const mlk::Matrix<T>&,
                                mlk::Matrix<T>&
                                );
