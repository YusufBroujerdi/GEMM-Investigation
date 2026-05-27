#include "mlkernels/validate.hpp"
#include "mlkernels/matrix.hpp"
#include <stdexcept>
#include <cmath>

namespace mlk {

template<typename T>
T max_abs_diff(Matrix<T> candidate, Matrix<T> reference) {

    Matrix<T> can = candidate, ref = reference;

    if (can.rows() != ref.rows() || can.cols() != ref.cols())
        throw std::invalid_argument("Matrices for validation have different dimensions");

    T max_diff = 0, diff;

    for (std::size_t i = 0; i < ref.rows(); i++)
        for (std::size_t j = 0; j < ref.cols(); j++)
            if ((diff = std::abs(ref(i, j) - can(i, j))) > max_diff)
                max_diff = diff;

    return max_diff;
}

template<typename T>
T max_rel_diff(Matrix<T> candidate, Matrix<T> reference) {

    Matrix<T> can = candidate, ref = reference;

    if (can.rows() != ref.rows() || can.cols() != ref.cols())
        throw std::invalid_argument("Matrices for validation have different dimensions");

    T max_diff = 0, diff;

    auto rel_equation = [](T ref, T can) {
        //We take the max between 1 and the ref value to ensure small values
        //don't warp the result, and the result remains defined for ref = 0
        return std::abs(ref - can) / std::max<T>(ref, 1);
    };

    for (std::size_t i = 0; i < ref.rows(); i++)
        for (std::size_t j = 0; j < ref.cols(); j++)
            if ((diff = rel_equation(ref(i, j), can(i, j))) > max_diff)
                max_diff = diff;

    return max_diff;
}


template<typename T>
T mean_abs_diff(Matrix<T> candidate, Matrix<T> reference) {

    Matrix<T> can = candidate, ref = reference;

    if (can.rows() != ref.rows() || can.cols() != ref.cols())
        throw std::invalid_argument("Matrices for validation have different dimensions");

    T sum = 0;

    for (std::size_t i = 0; i < ref.rows(); i++)
        for (std::size_t j = 0; j < ref.cols(); j++)
            sum += std::abs(ref(i, j) - can(i, j));

    return sum / (ref.rows() * ref.cols());
}

template <typename T>
T mean_rel_diff(Matrix<T> candidate, Matrix<T> reference) {

    Matrix<T> can = candidate, ref = reference;

    if (can.rows() != ref.rows() || can.cols() != ref.cols())
        throw std::invalid_argument("Matrices for validation have different dimensions");

    T sum = 0;

    auto rel_equation = [](T ref, T can) {
        //We take the max between 1 and the ref value to ensure small values
        //don't warp the result, and the result remains defined for ref = 0
        return std::abs(ref - can) / std::max<T>(ref, 1);
    };

    for (std::size_t i = 0; i < ref.rows(); i++)
        for (std::size_t j = 0; j < ref.cols(); j++)
            sum += rel_equation(ref(i, j), can(i, j));

    return sum / (ref.rows() * ref.cols());
}

template float  max_abs_diff<float> (Matrix<float>,  Matrix<float>);
template double max_abs_diff<double>(Matrix<double>, Matrix<double>);

template float  max_rel_diff<float> (Matrix<float>,  Matrix<float>);
template double max_rel_diff<double>(Matrix<double>, Matrix<double>);

template float  mean_abs_diff<float> (Matrix<float>,  Matrix<float>);
template double mean_abs_diff<double>(Matrix<double>, Matrix<double>);

template float  mean_rel_diff<float> (Matrix<float>,  Matrix<float>);
template double mean_rel_diff<double>(Matrix<double>, Matrix<double>);

}
