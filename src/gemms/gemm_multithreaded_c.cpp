
#include <stdexcept>

#include "mlkernels/gemm.hpp"

template<typename T>
void mlk::multithreaded_gemm_c(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    std::size_t num_threads
)
{
    if (left.cols() != right.rows())
        throw std::invalid_argument(
            "left.cols() must equal right.rows() for matrix multiplication."
        );

    if (left.rows() != output.rows()
        or right.cols() != output.cols())
        throw std::invalid_argument(
            "output matrix dimensions must be left.rows() by right.cols()."
        );

    if (num_threads < 1)
        throw std::invalid_argument(
            "n_threads must be at least 1."
        );

    output.fill(0);

    #pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
    for (std::size_t i = 0; i < left.rows(); i++)
        for (std::size_t j = 0; j < right.cols(); j++)
            for (std::size_t k = 0; k < left.cols(); k++)
                output(i, j) += left(i, k) * right(k, j);
}

template
void mlk::multithreaded_gemm_c<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output,
    std::size_t num_threads
);

template
void mlk::multithreaded_gemm_c<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output,
    std::size_t num_threads
);

