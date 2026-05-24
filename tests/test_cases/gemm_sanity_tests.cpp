#include <array>
#include "mlkernels/matrix.hpp"

namespace gemm_sanity_tests {
    
mlk::Matrix<float> left_1 {1, 3,
    {
    3.24, -2.1, 4.66
    }
};

mlk::Matrix<float> right_1{3, 1,
    {
        4.69,
        2.105,
        0.393
    }
};

mlk::Matrix<float> output_1{1, 1,
    {
        12.60648
    }
};

mlk::MatrixGemmTestCase<float> test_1 {left_1, right_1, output_1, "sanity_1"};

mlk::Matrix<float> left_2{2, 2,
    {
        3.5,	5.325,
        -3.6,	-13.3
    }
};

mlk::Matrix<float> right_2{2, 3,
    {
    4.69,	-2.33,	1.55,
    2.105,	 0.46,	-5.0
    }
};

mlk::Matrix<float> output_2{2, 3,
    {
        27.624125, -5.7055, -21.2,
         -44.8805,    2.27, 60.92
    }
};

mlk::MatrixGemmTestCase<float> test_2 {left_2, right_2, output_2, "sanity_2"};

mlk::Matrix<float> left_3{3, 5,
    {
        2,  1,  5,  14,  -3,
        4,  0, 55,  22, -56,
        4,  3,  2, -10,  34,
    }
};

mlk::Matrix<float> right_3{5, 3,
    {
         22,	  3,	 35,
          1,	 55,	-31,
         -5,	-32,	 33,
        -61,	 22,	 11,
          2,	 12,	 43
    }
};

mlk::Matrix<float> output_3{3, 3,
    {
         -840,	  173,	 229,
        -1641,	-1936,	-211,
          759,	  301,	1465
    }
};

mlk::MatrixGemmTestCase<float> test_3 {left_3, right_3, output_3, "sanity_3"};

std::array<mlk::MatrixGemmTestCase<float>, 3> tests {test_1, test_2, test_3};

}
