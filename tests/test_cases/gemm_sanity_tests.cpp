#include <array>
#include "mlkernels/matrix.hpp"

namespace gemm_sanity_tests {
    
mlk::Matrix<float> left_1 {1, 3,
    {
    3.24f, -2.1f, 4.66f
    }
};

mlk::Matrix<float> right_1{3, 1,
    {
        4.69f,
        2.105f,
        0.393f
    }
};

mlk::Matrix<float> output_1{1, 1,
    {
        12.60648f
    }
};

mlk::MatrixGemmTestCase<float> test_1 {left_1, right_1, output_1, "sanity_1"};

mlk::Matrix<float> left_2{2, 2,
    {
        3.5f,	5.325f,
        -3.6f,	-13.3f
    }
};

mlk::Matrix<float> right_2{2, 3,
    {
    4.69f,	-2.33f,	1.55f,
    2.105f,	 0.46f,	-5.0f
    }
};

mlk::Matrix<float> output_2{2, 3,
    {
        27.624125f, -5.7055f, -21.2f,
         -44.8805f,    2.27f, 60.92f
    }
};

mlk::MatrixGemmTestCase<float> test_2 {left_2, right_2, output_2, "sanity_2"};

mlk::Matrix<float> left_3{3, 5,
    {
        2.0f,  1.0f,  5.0f,  14.0f,  -3.0f,
        4.0f,  0.0f, 55.0f,  22.0f, -56.0f,
        4.0f,  3.0f,  2.0f, -10.0f,  34.0f,
    }
};

mlk::Matrix<float> right_3{5, 3,
    {
         22.0f,	  3.0f,	 35.0f,
          1.0f,	 55.0f,	-31.0f,
         -5.0f,	-32.0f,	 33.0f,
        -61.0f,	 22.0f,	 11.0f,
          2.0f,	 12.0f,	 43.0f
    }
};

mlk::Matrix<float> output_3{3, 3,
    {
         -840.0f,	  173.0f,	 229.0f,
        -1641.0f,	-1936.0f,	-211.0f,
          759.0f,	  301.0f,	1465.0f
    }
};

mlk::MatrixGemmTestCase<float> test_3 {left_3, right_3, output_3, "sanity_3"};

std::array<mlk::MatrixGemmTestCase<float>, 3> tests {test_1, test_2, test_3};

}
