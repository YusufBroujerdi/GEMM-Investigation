
#include <stdexcept>

#include "mlkernels/gemm.hpp"

template<typename T>
void mlk::reordered_gemm(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output
)
{
    //If i1, i2, i3, correspond to the row of left, right and output resp.
    //and j1, j2, j3, correspond to the col of left, right and output resp.
    //we have j1 = i2, i1 = i3 and j2 = j3
    std::size_t i, j, k;

    if (left.cols() != right.rows())
        throw std::invalid_argument(
            "reordered_gemm: left.cols() must equal right.rows() for matrix multiplication."
        );

    if (left.rows() != output.rows()
        or right.cols() != output.cols())
        throw std::invalid_argument(
            "reordered_gemm: output matrix dimensions must be left.rows() by right.cols()."
        );

    output.fill(0);

    for (i = 0; i < left.rows(); i++)
        for (k = 0; k < left.cols(); k++)
            for (j = 0; j < right.cols(); j++)
                output(i, j) += left(i, k) * right(k, j);
}

template
void mlk::reordered_gemm<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output
);

template
void mlk::reordered_gemm<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output
);
