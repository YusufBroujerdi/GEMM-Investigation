#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "mlkernels/matrix.hpp"
#include "mlkernels/gemm.hpp"


template<typename T> void mlk::multithreaded_gemm_a(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    std::size_t max_block_size,
    int n_threads
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

    if (n_threads < 1)
        throw std::invalid_argument(
            "n_threads must be at least 1."
        );

    output.fill(0);

    #pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2)
    for (std::size_t ib = 0; ib < output.rows(); ib += max_block_size)
        for (std::size_t jb = 0; jb < output.cols(); jb += max_block_size)
            for (std::size_t kb = 0; kb < left.cols(); kb += max_block_size)

            {
                //Check if our tile is too close to any edge and truncate it if necessary.
                std::size_t ib_endpoint = std::min(ib + max_block_size, output.rows());
                std::size_t kb_endpoint = std::min(kb + max_block_size, left.cols());
                std::size_t jb_endpoint = std::min(jb + max_block_size, output.cols());


                for (std::size_t i = ib; i < ib_endpoint; i++)
                    for (std::size_t k = kb; k < kb_endpoint; k++)
                        for (std::size_t j = jb; j < jb_endpoint; j++)
                            output(i, j) += left(i, k) * right(k, j);
            }

}

template
void mlk::multithreaded_gemm_a<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output,
    std::size_t max_block_size,
    int n_threads
);

template
void mlk::multithreaded_gemm_a<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output,
    std::size_t max_block_size,
    int n_threads
);
