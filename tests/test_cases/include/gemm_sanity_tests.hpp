#pragma once
#include <array>
#include "mlkernels/matrix.hpp"

namespace gemm_sanity_tests{

    extern std::array<mlk::MatrixGemmTestCase<float>, 3> tests;
}
