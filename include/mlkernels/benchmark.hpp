#pragma once
#include <filesystem>

class GemmBenchmark;
class BenchResult;

void benchmark(std::filesystem::path spec_csv, std::filesystem::path result_csv);
