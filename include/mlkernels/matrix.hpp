#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>

namespace mlk {

template <typename T>
class Matrix {

public:

    Matrix()
        : rows_(0), cols_(0), data_() {}

    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const {
        return cols_;
    }

    std::size_t size() const {
        return data_.size();
    }

    const T* data() const {
        return data_.data();
    }

    T* data() {
        return data_.data();
    }

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

}
