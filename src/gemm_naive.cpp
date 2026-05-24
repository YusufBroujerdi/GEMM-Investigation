#include "mlkernels/gemm.hpp"

template<typename T>
int mlk::naive_gemm(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output
)
{
    //If i1, i2, i3, correspond to the row of left, right and output resp.
    //and j1, j2, j3, correspond to the col of left, right and output resp.
    //we have j1 = i2, i1 = i3 and j2 = j3
    std::size_t i1, j1, j2;
    
    //Check left and right are compatible for multiplication.
    if (left.cols() != right.rows())
        return 1;

    //Check output matrix has the right dimensions.
    if (left.rows() != output.rows()
        or right.cols() != output.cols())
        return 2;

    output.fill(0);

    //Take the dot product of the left row and the right col,
    //for each value in the output matrix
    for (j2 = 0; j2 < right.cols(); j2++) 
        for (i1 = 0; i1 < left.rows(); i1++) 
            for (j1 = 0; j1 < left.cols(); j1++) 
                output(i1, j2) += left(i1, j1) * right(j1, j2);    
    
    return 0;
}

template
int mlk::naive_gemm<float>(
    const mlk::Matrix<float>& left,
    const mlk::Matrix<float>& right,
    mlk::Matrix<float>& output
);

template
int mlk::naive_gemm<double>(
    const mlk::Matrix<double>& left,
    const mlk::Matrix<double>& right,
    mlk::Matrix<double>& output
);
