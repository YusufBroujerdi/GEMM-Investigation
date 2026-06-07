#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "mlkernels/matrix.hpp"
#include "mlkernels/gemm.hpp"

template<typename T>
void mlk::tiled_gemm_b(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    std::size_t max_block_size
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

    tiled_gemm_b(left, right, output, tilestate);
}

template<typename T>
void mlk::tiled_gemm_b(
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
        tiled_gemm_b(left, right, output, ts);

        ts.xpos += ts.width;
        ts.width += remainder;
        tiled_gemm_b(left, right, output, ts);

        ts.xpos -= ts.width - remainder;
        ts.width = ts.width * 2 - remainder;
        return;
    }

    if (ts.height == largest) {

        int remainder = ts.height % 2;
        ts.height /= 2;
        tiled_gemm_b(left, right, output, ts);

        ts.ypos += ts.height;
        ts.height += remainder;
        tiled_gemm_b(left, right, output, ts);

        ts.ypos -= ts.height - remainder;
        ts.height = ts.height * 2 - remainder;
        return;
    }

    if (ts.ksize == largest) {

        int remainder = ts.ksize % 2;
        ts.ksize /= 2;
        tiled_gemm_b(left, right, output, ts);

        ts.kpos += ts.ksize;
        ts.ksize += remainder;
        tiled_gemm_b(left, right, output, ts);

        ts.kpos -= ts.ksize - remainder;
        ts.ksize = ts.ksize * 2 - remainder;
        return;
    }
}


template
void mlk::tiled_gemm_b<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output,
    std::size_t max_block_size
);

template
void mlk::tiled_gemm_b<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output,
    std::size_t max_block_size
);
