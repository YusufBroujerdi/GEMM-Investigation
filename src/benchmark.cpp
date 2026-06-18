#include "mlkernels/matrix.hpp"
#include "mlkernels/gemm.hpp"
#include "mlkernels/benchmark.hpp"
#include "mlkernels/validate.hpp"
#include <string>
#include <array>
#include <queue>
#include <map>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <type_traits>
#include <iostream>













constexpr std::uint16_t benchmark_schema_size = 10;
constexpr std::uint16_t number_of_gemms = 7;

std::map<std::string, mlk::GemmKernels> kernel_lookup{
    {"Kernel.naive" , mlk::GemmKernels::Naive},
    {"Kernel.reordered", mlk::GemmKernels::Reordered},
    {"Kernel.tiled_a", mlk::GemmKernels::Tiled_a},
    {"Kernel.tiled_b", mlk::GemmKernels::Tiled_b},
    {"Kernel.multithreaded_a", mlk::GemmKernels::Multithreaded_a},
    {"Kernel.multithreaded_b", mlk::GemmKernels::Multithreaded_b},
    {"Kernel.multithreaded_c", mlk::GemmKernels::Multithreaded_c}
};

std::array<std::string, number_of_gemms> kernel_str_lookup = {
    "gemm_naive",
    "gemm_reordered",
    "gemm_tiled_a",
    "gemm_tiled_b",
    "gemm_multithreaded_a",
    "gemm_multithreaded_b",
    "gemm_multithreaded_c"
};

template <typename T>
using BenchmarkGemmFuncPtr = void (*)(
    const mlk::Matrix<T>&,
    const mlk::Matrix<T>&,
    mlk::Matrix<T>&,
    const GemmBenchmark&
);

template <typename T>
void naive_gemm_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark&
) { return mlk::naive_gemm(left, right, output); };

template <typename T>
void reordered_gemm_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark&
) { return mlk::reordered_gemm(left, right, output); };

template <typename T>
void tiled_gemm_a_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark& schema
) { return mlk::tiled_gemm_a(left, right, output, schema.block_size()); };

template <typename T>
void tiled_gemm_b_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark& schema
) { return mlk::tiled_gemm_b(left, right, output, schema.block_size()); };

template <typename T>
void multithreaded_gemm_a_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark& schema
) { return mlk::multithreaded_gemm_a(left, right, output, schema.block_size(), schema.threads()); };

template <typename T>
void multithreaded_gemm_b_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark& schema
) { return mlk::multithreaded_gemm_b(left, right, output, schema.block_size(), schema.threads()); };

template <typename T>
void multithreaded_gemm_c_wrapped (
    const mlk::Matrix<T>& left,
    const mlk::Matrix<T>& right,
    mlk::Matrix<T>& output,
    const GemmBenchmark& schema
) { return mlk::multithreaded_gemm_c(left, right, output, schema.threads()); };


template <typename T>
std::array<BenchmarkGemmFuncPtr<T>, number_of_gemms> kernel_func_lookup = {
    naive_gemm_wrapped,
    reordered_gemm_wrapped,
    tiled_gemm_a_wrapped,
    tiled_gemm_b_wrapped,
    multithreaded_gemm_a_wrapped,
    multithreaded_gemm_b_wrapped,
    multithreaded_gemm_c_wrapped
};

std::map<std::string, mlk::FloatTypes> float_lookup{
    {"FloatType.float_t", mlk::FloatTypes::Float},
    {"FloatType.double_t", mlk::FloatTypes::Double}
};

std::array<std::string, 2> float_str_lookup = {"float", "double"};

template <typename Enum>
std::size_t to_index(Enum e) {
    return static_cast<std::size_t>(e);
}
















GemmBenchmark::GemmBenchmark(std::string schema) {

    std::string::size_type index = 0;
    std::string::size_type new_index = 0;
    std::uint16_t schema_index = 0;
    std::array<std::string, benchmark_schema_size> split_schema;

    for (schema_index = 0; schema_index < benchmark_schema_size; schema_index++) {

        new_index = schema.find(",", index);
        split_schema[schema_index] = schema.substr(index, new_index - index);
        if (new_index == std::string::npos && schema_index != 9)
            throw std::invalid_argument("csv has wrong number of columns");
        if (split_schema[schema_index] == "")
            throw std::invalid_argument("field in csv empty");
        index = ++new_index;
    }

    case_name_ = split_schema[0];
    m_ = std::stoi(split_schema[1]);
    k_ = std::stoi(split_schema[2]);
    n_ = std::stoi(split_schema[3]);
    float_type_ = float_lookup.at(split_schema[4]);
    kernel_ = kernel_lookup.at(split_schema[5]);
    block_size_ = std::stoi(split_schema[6]);
    threads_ = std::stoi(split_schema[7]);
    repetitions_ = std::stoi(split_schema[8]);
    seed_ = std::stoi(split_schema[9]);
}


const std::string& GemmBenchmark::case_name() const { return case_name_; }
std::size_t GemmBenchmark::m() const { return m_; }
std::size_t GemmBenchmark::k() const { return k_; }
std::size_t GemmBenchmark::n() const { return n_; }
mlk::FloatTypes GemmBenchmark::float_type() const { return float_type_; }
mlk::GemmKernels GemmBenchmark::kernel() const { return kernel_; }
std::size_t GemmBenchmark::block_size() const { return block_size_; }
std::uint64_t GemmBenchmark::threads() const { return threads_; }
std::uint64_t GemmBenchmark::repetitions() const { return repetitions_; }
std::uint64_t GemmBenchmark::seed() const { return seed_; }


BenchResult::BenchResult(
    GemmBenchmark original_benchmark,
    std::chrono::duration<double, std::milli> time_ms_min,
    std::chrono::duration<double, std::milli> time_ms_max,
    std::chrono::duration<double, std::milli> time_ms_mean,
    double gflops_per_second,
    double max_abs_error,
    double max_rel_error,
    double mean_abs_error,
    bool validation_result
) : original_benchmark_(original_benchmark), time_ms_min_(time_ms_min),
    time_ms_max_(time_ms_max), time_ms_mean_(time_ms_mean),
    gflops_per_second_(gflops_per_second), max_abs_error_(max_abs_error),
    max_rel_error_(max_rel_error), mean_abs_error_(mean_abs_error),
    validation_result_(validation_result) {}

const GemmBenchmark& BenchResult::original_benchmark() const { return original_benchmark_; }
std::chrono::duration<double, std::milli> BenchResult::time_ms_min() const { return time_ms_min_; }
std::chrono::duration<double, std::milli> BenchResult::time_ms_max() const { return time_ms_max_; }
std::chrono::duration<double, std::milli> BenchResult::time_ms_mean() const { return time_ms_mean_; }
double BenchResult::gflops_per_second() const { return gflops_per_second_; }
double BenchResult::max_abs_error() const { return max_abs_error_; }
double BenchResult::max_rel_error() const { return max_rel_error_; }
double BenchResult::mean_abs_error() const { return mean_abs_error_; }
bool BenchResult::validation_result() const { return validation_result_; }


void write_benchresult(BenchResult& result, std::ofstream& file) {

    GemmBenchmark bm = result.original_benchmark();

    file << bm.case_name() << "," << bm.m() << "," << bm.k() << "," << bm.n()
        << "," << float_str_lookup[to_index(bm.float_type())]
        << "," << kernel_str_lookup[to_index(bm.kernel())] << ","
        << bm.block_size() << "," << bm.threads() << "," << bm.repetitions() << ","
        << bm.seed() << ",";

    file << result.time_ms_min().count() << "," << result.time_ms_max().count() << ","
        << result.time_ms_mean().count() << "," << result.gflops_per_second() << ","
        << result.max_abs_error() << "," << result.max_rel_error() << ","
        << result.mean_abs_error() << "," << result.validation_result() << '\n';

}













template<typename T>
BenchResult benchmark_templated(GemmBenchmark spec) {

    using clock = std::chrono::steady_clock;
    using dms = std::chrono::duration<double, std::milli>;
    dms min{dms::max()}, max{0}, sum{0};
    std::mt19937_64 gen{spec.seed()};

    std::cout << "Beginning test on " << spec.case_name() << "\n";

    mlk::GemmTestCase<T> test_case {spec.m(), spec.k(), spec.n(), gen, spec.case_name()};
    mlk::Matrix<T> candidate {spec.m(), spec.n()};
    BenchmarkGemmFuncPtr<T> gemm = kernel_func_lookup<T>[to_index(spec.kernel())];

    //Warm-up / validation
    gemm(test_case.left(), test_case.right(), candidate, spec);
    T max_abs_diff = mlk::max_abs_diff(candidate, test_case.output());
    T max_rel_diff = mlk::max_rel_diff(candidate, test_case.output());
    T mean_abs_diff = mlk::mean_abs_diff(candidate, test_case.output());
    bool validation_result = max_abs_diff < mlk::tolerance<T>;
    gemm(test_case.left(), test_case.right(), candidate, spec);

    std::cout << "Warm-up complete\n";

    for (std::uint64_t i = 0; i < spec.repetitions(); i++) {

        auto begin = clock::now();
        gemm(test_case.left(), test_case.right(), candidate, spec);
        auto end = clock::now();

        dms duration = end - begin;
        min = duration < min ? duration : min;
        max = duration > max ? duration : max;
        sum += duration;

        std::cout << "Iteration " << i + 1 << " complete\n";
    }

    double computations = 2.0 * spec.m() * spec.n() * spec.k();
    double gflops = computations * spec.repetitions() / sum.count() / 1e6;
    dms mean = sum / spec.repetitions();

    std::cout << "Test case successfully complete. Results written to file\n";

    return {spec, min, max, mean, gflops,
        max_abs_diff, max_rel_diff, mean_abs_diff, validation_result};
}
        

BenchResult benchmark(GemmBenchmark spec) {

    switch (spec.float_type()) {

        case mlk::FloatTypes::Float:
            return benchmark_templated<float>(spec);

        case mlk::FloatTypes::Double:
            return benchmark_templated<double>(spec);
    }

    throw std::logic_error("Failed to recognize float type");
}


void benchmark(std::filesystem::path spec_csv, std::filesystem::path result_csv) {

    auto parent = result_csv.parent_path();
    if (!std::filesystem::exists(parent))
        std::filesystem::create_directory(parent);

    std::ifstream specs{spec_csv};

    if (!specs)
        throw std::invalid_argument("spec csv file error.");

    std::queue<GemmBenchmark> spec_list;
    std::queue<BenchResult> res_list;

    // load the benchmark specs (the first row of the csv is a header)
    std::string line;
    std::getline(specs, line);
    while (std::getline(specs, line))
        spec_list.emplace(GemmBenchmark(line));

    while (!spec_list.empty()) {
        res_list.push(benchmark(spec_list.front()));
        spec_list.pop();
    }

    std::ofstream result{result_csv};

    result << "case_name,m,k,n,float_type,kernel,block_size,threads,repetitions,seed,";
    result << "time_ms_min,time_ms_max,time_ms_mean,gflops_per_second,";
    result << "max_abs_error,max_rel_error,mean_abs_error,validation_result\n";
    while (!res_list.empty()) {
        write_benchresult(res_list.front(), result);
        res_list.pop();
    }
}
