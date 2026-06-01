#pragma once
#include <filesystem>
#include <cstdint>
#include "mlkernels/matrix.hpp"
#include "mlkernels/gemm.hpp"

class BenchResult;

class GemmBenchmark
{

public:

    GemmBenchmark(std::string schema);
    const std::string& case_name() const;
    std::size_t m() const;
    std::size_t k() const;
    std::size_t n() const;
    mlk::FloatTypes float_type() const;
    mlk::GemmKernels kernel() const; 
    std::size_t block_size() const; 
    std::uint64_t threads() const; 
    std::uint64_t repetitions() const; 
    std::uint64_t seed() const; 

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
    );
    
    const GemmBenchmark& original_benchmark() const; 
    std::chrono::milliseconds time_ms_min() const; 
    std::chrono::milliseconds time_ms_max() const; 
    std::chrono::duration<double, std::milli> time_ms_mean() const; 
    double gflops_per_second() const; 
    double max_abs_error() const; 
    double max_rel_error() const; 
    double mean_abs_error() const; 
    bool validation_result() const; 

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

void benchmark(std::filesystem::path spec_csv, std::filesystem::path result_csv);
