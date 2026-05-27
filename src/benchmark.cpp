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


constexpr std::uint16_t benchmark_schema_size = 10;

std::map<std::string, mlk::GemmKernels> kernel_lookup{
    {"gemm_naive" , mlk::GemmKernels::Naive},
    {"gemm_reordered", mlk::GemmKernels::Reordered}
};

std::array<std::string, 2> kernel_str_lookup = {"gemm_naive", "gemm_reordered"};

template<typename T>
std::array<GemmFunctionPtr<T>, 2> kernel_func_lookup = {
    mlk::naive_gemm,
    mlk::reordered_gemm
};

std::map<std::string, mlk::FloatTypes> float_lookup{
    {"float", mlk::FloatTypes::Float},
    {"double", mlk::FloatTypes::Double}
};

std::array<std::string, 2> float_str_lookup = {"float", "double"};

template <typename Enum>
std::size_t to_index(Enum e) {
    return static_cast<std::size_t>(e);
}


class GemmBenchmark
{

public:

    GemmBenchmark(std::string schema) {

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

    const std::string& case_name() const { return case_name_; }

    std::size_t m() const { return m_; }

    std::size_t k() const { return k_; }

    std::size_t n() const { return n_; }

    mlk::FloatTypes float_type() const { return float_type_; }

    mlk::GemmKernels kernel() const { return kernel_; }

    std::size_t block_size() const { return block_size_; }

    std::uint64_t threads() const { return threads_; }

    std::uint64_t repetitions() const { return repetitions_; }

    std::uint64_t seed() const { return seed_; }

private:

    std::string case_name_;
    std::size_t m_;
    std::size_t k_;
    std::size_t n_;
    mlk::FloatTypes float_type_;
    mlk::GemmKernels kernel_;
    std::size_t block_size_;
    std::uint64_t threads_;
    std::uint64_t repetitions_;
    std::uint64_t seed_;

};


void read_benchmarks(std::queue<GemmBenchmark>& benchmarks, std::filesystem::path path) {

    std::string line;
    std::ifstream file{path};

    if (!file)
        throw std::invalid_argument("Invalid file address");

    std::getline(file, line);
    while (std::getline(file, line))
        benchmarks.emplace(GemmBenchmark(line));
}


class BenchResult {

public:

    BenchResult(
        GemmBenchmark original_benchmark,
        std::chrono::milliseconds time_ms_min,
        std::chrono::milliseconds time_ms_max,
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

    const GemmBenchmark& original_benchmark() const { return original_benchmark_; }

    std::chrono::milliseconds time_ms_min() const { return time_ms_min_; }

    std::chrono::milliseconds time_ms_max() const { return time_ms_max_; }

    std::chrono::duration<double, std::milli> time_ms_mean() const { return time_ms_mean_; }

    double gflops_per_second() const { return gflops_per_second_; }

    double max_abs_error() const { return max_abs_error_; }

    double max_rel_error() const { return max_rel_error_; }

    double mean_abs_error() const { return mean_abs_error_; }

    bool validation_result() const { return validation_result_; }

private:

    GemmBenchmark original_benchmark_;
    std::chrono::milliseconds time_ms_min_;
    std::chrono::milliseconds time_ms_max_;
    std::chrono::duration<double, std::milli> time_ms_mean_;
    double gflops_per_second_;
    double max_abs_error_;
    double max_rel_error_;
    double mean_abs_error_;
    bool validation_result_;

};


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


void write_benchresults(std::queue<BenchResult>& results, std::filesystem::path path) {

    std::ofstream file{path};

    if (!file)
        throw std::invalid_argument("Invalid file address");
    
    while (!results.empty()) {
        write_benchresult(results.front(), file);
        results.pop();
    }
}


template<typename T>
BenchResult benchmark_templated(GemmBenchmark spec) {

    using clock = std::chrono::steady_clock;
    using ms = std::chrono::milliseconds;
    ms min{ms::max()}, max{0}, sum{0};
    std::mt19937_64 gen{spec.seed()};

    mlk::GemmTestCase<T> test_case {spec.m(), spec.k(), spec.n(), gen, spec.case_name()};
    mlk::Matrix<T> candidate {spec.m(), spec.n()};
    GemmFunctionPtr<T> gemm = kernel_func_lookup<T>[to_index(spec.kernel())];

    //Warm-up / validation
    gemm(test_case.left(), test_case.right(), candidate);
    T max_abs_diff = mlk::max_abs_diff(candidate, test_case.output());
    T max_rel_diff = mlk::max_rel_diff(candidate, test_case.output());
    T mean_abs_diff = mlk::mean_abs_diff(candidate, test_case.output());
    bool validation_result = max_abs_diff < mlk::tolerance<T>;
    gemm(test_case.left(), test_case.right(), candidate);

    for (std::uint64_t i = 0; i < spec.repetitions(); i++) {

        auto begin = clock::now();
        gemm(test_case.left(), test_case.right(), candidate);
        auto end = clock::now();

        auto duration = std::chrono::duration_cast<ms>(end - begin);
        min = duration < min ? duration : min;
        max = duration > max ? duration : max;
        sum += duration;
    }

    double computations = static_cast<double>(2 * spec.m() * spec.n() * spec.k());
    double gflops = computations / sum.count() / 1e6;
    std::chrono::duration<double, std::milli> mean = sum / spec.repetitions();

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
}
