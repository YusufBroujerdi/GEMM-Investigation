#pragma once

#include <cstdint>

#include "mlkernels/matrix.hpp"


namespace mlk {

struct TileState{
    //This object tracks the dimensions of the sub-matrix being operated on in one place
    //in order to simplify function calls (and hopefully reduce function call overhead).
    std::size_t xpos; // Tracks xpos of top-left of output matrix tile
    std::size_t ypos; // Tracks ypos of top-left of output matrix tile
    std::size_t kpos; // Tracks xpos of left matrix tile / ypos of right matrix tile
    std::size_t width; // Tracks width of output matrix tile
    std::size_t height; // Tracks height of output matrix tile
    std::size_t ksize; // Tracks width of left matrix tile / hegiht of right matrix tile
    std::size_t max_block_size; // Tracks max block size in one place
};

template<typename T>
void tiled_gemm_b(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    TileState& ts
);

template<typename T>
void multithreaded_gemm_b(
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    TileState& ts
);

}
