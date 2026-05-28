#include "mlkernels/gemm.hpp"
#include "mlkernels/benchmark.hpp"
#include <exception>

int main(int argc, char *argv[]) {

    if (argc != 3)
        throw std::invalid_argument("Must supply exactly 2 paths");

    std::string spec_csv = argv[1];
    std::string result_csv = argv[2];

    benchmark(spec_csv, result_csv);
    return 0;
}
