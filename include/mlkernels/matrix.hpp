#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace mlk {

template <typename T>
class Matrix {

public:

    Matrix()
        : rows_(0), cols_(0), data_() {}

    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    template <std::size_t N>
    Matrix(std::size_t rows, std::size_t cols, const T (&values)[N])
        : Matrix(rows, cols) {

        if(rows * cols != N)
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

private:

    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

template <typename T>
class MatrixGemmTestCase {

public:

    MatrixGemmTestCase(Matrix<T> left, Matrix<T> right, Matrix<T> output)
        : left_(left), right_(right), output_(output), name_("generic") {}

    MatrixGemmTestCase(Matrix<T> left,
                   Matrix<T> right,
                   Matrix<T> output,
                   std::string name)
        : left_(left), right_(right), output_(output), name_(name) {}

    const std::string& name() const { return name_; }

    const Matrix<T>& left() const { return left_; }

    const Matrix<T>& right() const {return right_; }

    const Matrix<T>& output() const {return output_; }

private:

    Matrix<T> left_;
    Matrix<T> right_;
    Matrix<T> output_;
    std::string name_;
};

}
