#pragma once

#include "mlkernels/matrix.hpp"

namespace mlk {

enum class validation {
    Passed = 0,
    Failed = 1
};

template <typename T>
T max_abs_diff(const Matrix<T>& candidate, const Matrix<T>& reference);

template <typename T>
T max_rel_diff(const Matrix<T>& candidate, const Matrix<T>& reference);

template <typename T>
T mean_abs_diff(const Matrix<T>& candidate, const Matrix<T>& reference);

template <typename T>
T mean_rel_diff(const Matrix<T>& candidate, const Matrix<T>& reference);

}
