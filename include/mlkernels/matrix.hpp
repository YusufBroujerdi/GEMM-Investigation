#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace mlk {


enum class FloatTypes {
    Float = 0,
    Double = 1,
};


template <typename T>
class Matrix {

public:

    Matrix();
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, std::mt19937_64& gen);
    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<T> values);
    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t size() const;
    const T* data() const;
    T* data();
    const T& operator()(std::size_t row, std::size_t col) const;
    T& operator()(std::size_t row, std::size_t col);
    void fill(T value);
    void write(std::ostream& output) const;
    void randomize(std::mt19937_64& gen);

private:

    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;

};


template <typename T>
class GemmTestCase {

public:

    GemmTestCase(
        Matrix<T> left,
        Matrix<T> right,
        Matrix<T> output
    );
    GemmTestCase(
        Matrix<T> left,
        Matrix<T> right,
        Matrix<T> output,
        std::string name
    );
    GemmTestCase(
        std::size_t M,
        std::size_t K,
        std::size_t N,
        std::mt19937_64& gen,
        std::string name
    );
    GemmTestCase(
        std::size_t M,
        std::size_t K,
        std::size_t N,
        std::mt19937_64& gen
    );

    const std::string& name() const;
    const Matrix<T>& left() const;
    const Matrix<T>& right() const;
    const Matrix<T>& output() const;
    void write(std::ostream& output) const;
    void randomize(std::mt19937_64& gen); 

private:

    Matrix<T> left_;
    Matrix<T> right_;
    Matrix<T> output_;
    std::string name_;
};

}
