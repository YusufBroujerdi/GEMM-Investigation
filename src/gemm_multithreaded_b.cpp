#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "mlkernels/matrix.hpp"
#include "mlkernels/gemm.hpp"

template<typename T>
void mlk::multithreaded_gemm_b(
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

    TileState tilestate{
        0,
        0,
        0,
        output.cols(),
        output.rows(),
        left.cols(),
        max_block_size
    };

    //Spin up a team of n_threads workers; a single thread initiates the recursion
    //and the rest pick up tasks from the queue as they are spawned.
    #pragma omp parallel num_threads(n_threads) default(none) shared(left, right, output, tilestate)
    {
        #pragma omp single
        multithreaded_gemm_b(left, right, output, tilestate);
    }
}

template<typename T>
void mlk::multithreaded_gemm_b(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    TileState& ts
)
{

    std::size_t largest = std::max({ts.width, ts.height, ts.ksize, ts.max_block_size});

    if (ts.max_block_size == largest) {
        for (std::size_t i = ts.ypos; i < ts.ypos + ts.height; i++)
            for (std::size_t k = ts.kpos; k < ts.kpos + ts.ksize; k++)
                for (std::size_t j = ts.xpos; j < ts.xpos + ts.width; j++)
                    output(i, j) += left(i, k) * right(k, j);
        return;
    }

    if (ts.width == largest) {

        int remainder = ts.width % 2;
        ts.width /= 2;
        #pragma omp task default(none) shared(left, right, output) firstprivate(ts)
        multithreaded_gemm_b(left, right, output, ts);

        ts.xpos += ts.width;
        ts.width += remainder;
        #pragma omp task default(none) shared(left, right, output) firstprivate(ts)
        multithreaded_gemm_b(left, right, output, ts);

        #pragma omp taskwait

        ts.xpos -= ts.width - remainder;
        ts.width = ts.width * 2 - remainder;
        return;
    }

    if (ts.height == largest) {

        int remainder = ts.height % 2;
        ts.height /= 2;
        #pragma omp task default(none) shared(left, right, output) firstprivate(ts)
        multithreaded_gemm_b(left, right, output, ts);

        ts.ypos += ts.height;
        ts.height += remainder;
        #pragma omp task default(none) shared(left, right, output) firstprivate(ts)
        multithreaded_gemm_b(left, right, output, ts);

        #pragma omp taskwait

        ts.ypos -= ts.height - remainder;
        ts.height = ts.height * 2 - remainder;
        return;
    }

    if (ts.ksize == largest) {

        //Parallelizing over k can cause data races without care. So I've left this sequential.
        int remainder = ts.ksize % 2;
        ts.ksize /= 2;
        multithreaded_gemm_b(left, right, output, ts);

        ts.kpos += ts.ksize;
        ts.ksize += remainder;
        multithreaded_gemm_b(left, right, output, ts);

        ts.kpos -= ts.ksize - remainder;
        ts.ksize = ts.ksize * 2 - remainder;
        return;
    }
}


template
void mlk::multithreaded_gemm_b<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output,
    std::size_t max_block_size,
    int n_threads
);

template
void mlk::multithreaded_gemm_b<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output,
    std::size_t max_block_size,
    int n_threads
);
