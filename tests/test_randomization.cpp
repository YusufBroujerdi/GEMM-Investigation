#include <random>
#include <fstream>
#include <filesystem>

#include <mlkernels/matrix.hpp>
#include <mlkernels/gemm.hpp>
#include <mlkernels/validate.hpp>

std::uint64_t seed_1 = 222;

std::uint64_t M1a = 2;
std::uint64_t N1a = 2;

std::uint64_t M1b = 2;
std::uint64_t N1b = 2;

std::uint64_t M1c = 3;
std::uint64_t N1c = 3;

std::uint64_t seed_2 = 64342;
std::uint64_t M2a = 2;
std::uint64_t N2a = 2;
std::uint64_t K2a = 2;

std::uint64_t M2b = 2;
std::uint64_t N2b = 2;

std::uint64_t M2c = 2;
std::uint64_t N2c = 3;
std::uint64_t K2c = 3;

std::uint64_t seed_3 = 221;

std::uint64_t M3a = 4;
std::uint64_t N3a = 4;
std::uint64_t K3a = 2;

std::uint64_t M3b = 3;
std::uint64_t N3b = 3;
std::uint64_t K3b = 3;
 
int main() {

    std::ofstream file {"local.txt"};

    std::mt19937_64 gen{seed_1};

    mlk::Matrix<float> matrix_1a{M1a, N1a, gen};
    matrix_1a.write(file);

    mlk::Matrix<float> matrix_1b{M1b, N1b, gen};
    matrix_1b.write(file);

    mlk::Matrix<float> matrix_1c{M1c, N1c, gen};
    matrix_1c.write(file);

    gen.seed(seed_2);

    mlk::GemmTestCase<float> testcase_2a{M2a, K2a, N2a, gen};
    testcase_2a.write(file);

    mlk::Matrix<float> matrix_2b{M2b, N2b, gen};
    matrix_2b.write(file);

    mlk::GemmTestCase<float> testcase_2c{M2c, K2c, N2c, gen};
    testcase_2c.write(file);

    gen.seed(seed_3);

    mlk::GemmTestCase<float> testcase_3a{M3a, K3a, N3a, gen};
    testcase_3a.write(file);

    mlk::GemmTestCase<float> testcase_3b{M3b, K3b, N3b, gen};
    testcase_3b.write(file);

    return 0;
}
