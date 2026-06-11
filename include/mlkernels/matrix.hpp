#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>
#include <stdexcept>
#include <random>

namespace mlk {


enum class FloatTypes {
    Float = 0,
    Double = 1,
};


template <typename T>
class Matrix {

public:

    Matrix()
        : rows_(0), cols_(0), data_() {}

    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    Matrix(std::size_t rows, std::size_t cols, std::mt19937_64& gen)
        : Matrix(rows, cols) {

            std::normal_distribution<float> normal{0.0, 100.0};

            for (T& value : data_)
                value = normal(gen);
        }

    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<T> values)
        : Matrix(rows, cols) {

        if(rows * cols != values.size())
            throw std::invalid_argument(
                "Number of values do not match matrix size."
            );

        std::copy(std::begin(values), std::end(values), data());
    }

    std::size_t rows() const { return rows_; }

    std::size_t cols() const { return cols_; }

    std::size_t size() const { return data_.size(); }

    const T* data() const { return data_.data(); }

    T* data() { return data_.data(); }

    const T& operator()(std::size_t row, std::size_t col) const
    {
        return data_[row * cols_ + col];
    }

    T& operator()(std::size_t row, std::size_t col)
    {
        return data_[row * cols_ + col];
    }

    void fill(T value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    void write(std::ostream& output) const {

        std::size_t i, j;

        output << "Matrix values: [\n";
        for (i = 0; i < rows_; i++) {
            for (j = 0; j < cols_; j++)
                output << std::fixed << std::setprecision(10)
                    << std::setw(18) << (*this)(i, j) << " ";
            output << "\n";
        }
        output << "]\n";
    }

    void randomize(std::mt19937_64& gen) {

        std::normal_distribution<float> normal{0.0, 100.0};

        for (T& value : data_)
            value = normal(gen);
    }

private:

    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};


template <typename T>
class GemmTestCase {

public:

    GemmTestCase(Matrix<T> left, Matrix<T> right, Matrix<T> output)
        : left_(left), right_(right), output_(output), name_("generic") {}

    GemmTestCase(Matrix<T> left,
                   Matrix<T> right,
                   Matrix<T> output,
                   std::string name)
        : left_(left), right_(right), output_(output), name_(name) {}

    GemmTestCase(std::size_t M,
                   std::size_t K,
                   std::size_t N,
                   std::mt19937_64& gen,
                   std::string name)
    : left_(M, K, gen), right_ (K, N, gen), output_(M, N), name_(name) {

        reordered_gemm(left_, right_ , output_);
    }

    GemmTestCase(std::size_t M,
                 std::size_t K,
                 std::size_t N,
                 std::mt19937_64& gen)
    : GemmTestCase(M, K, N, gen, "Generic_Random") {}

    const std::string& name() const { return name_; }

    const Matrix<T>& left() const { return left_; }

    const Matrix<T>& right() const {return right_; }

    const Matrix<T>& output() const {return output_; }

    void write(std::ostream& output) const {

        output << "Left ";
        left_.write(output);
        output << "Right ";
        right_.write(output);
        output << "Output ";
        output_.write(output);
    }

    void randomize(std::mt19937_64& gen) {

        left_.randomize(gen);
        right_.randomize(gen);
        output_.randomize(gen);
    }

private:

    Matrix<T> left_;
    Matrix<T> right_;
    Matrix<T> output_;
    std::string name_;
};

}
