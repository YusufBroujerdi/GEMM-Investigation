#include <stdexcept>

#include "mlkernels/gemm.hpp"

template<typename T>
void mlk::naive_gemm(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output
)
{
    std::size_t i, j, k;

    if (left.cols() != right.rows())
        throw std::invalid_argument(
            "left.cols() must equal right.rows() for matrix multiplication."
        );

    if (left.rows() != output.rows()
        or right.cols() != output.cols())
        throw std::invalid_argument(
            "output matrix dimensions must be left.rows() by right.cols()."
        );

    output.fill(0);

    for (j = 0; j < right.cols(); j++)
        for (i = 0; i < left.rows(); i++)
            for (k = 0; k < left.cols(); k++)
                output(i, j) += left(i, k) * right(k, j);
}

template
void mlk::naive_gemm<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output
);

template
void mlk::naive_gemm<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output
);
