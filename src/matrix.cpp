#include "mlkernels/matrix.hpp"
#include "mlkernels/gemm.hpp"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

template<typename T>
mlk::Matrix<T>::Matrix()
    : rows_(0), cols_(0), data_() {}

template<typename T>
mlk::Matrix<T>::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {}

template<typename T>
mlk::Matrix<T>::Matrix(
    std::size_t rows,
    std::size_t cols,
    std::mt19937_64& gen
) : Matrix(rows, cols) {

        std::normal_distribution<float> normal{0.0, 100.0};

        for (T& value : data_)
            value = normal(gen);
    }

template<typename T>
mlk::Matrix<T>::Matrix(
    std::size_t rows,
    std::size_t cols,
    std::initializer_list<T> values
) : Matrix(rows, cols) {

    if(rows * cols != values.size())
        throw std::invalid_argument(
            "Number of values do not match matrix size."
        );

    std::copy(std::begin(values), std::end(values), data());
}

template<typename T>
std::size_t mlk::Matrix<T>::rows() const { return rows_; }

template<typename T>
std::size_t mlk::Matrix<T>::cols() const { return cols_; }

template<typename T>
std::size_t mlk::Matrix<T>::size() const { return data_.size(); }

template<typename T>
const T* mlk::Matrix<T>::data() const { return data_.data(); }

template<typename T>
T* mlk::Matrix<T>::data() { return data_.data(); }

template<typename T>
const T& mlk::Matrix<T>::operator()(std::size_t row, std::size_t col) const
{
    return data_[row * cols_ + col];
}

template<typename T>
T& mlk::Matrix<T>::operator()(std::size_t row, std::size_t col)
{
    return data_[row * cols_ + col];
}

template<typename T>
void mlk::Matrix<T>::fill(T value)
{
    std::fill(data_.begin(), data_.end(), value);
}

template<typename T>
void mlk::Matrix<T>::write(std::ostream& output) const {

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

template<typename T>
void mlk::Matrix<T>::randomize(std::mt19937_64& gen) {

    std::normal_distribution<float> normal{0.0, 100.0};

    for (T& value : data_)
        value = normal(gen);
}



template<typename T>
mlk::GemmTestCase<T>::GemmTestCase(
    mlk::Matrix<T> left,
    mlk::Matrix<T> right,
    mlk::Matrix<T> output
) : left_(left), right_(right), output_(output), name_("generic") {}

template<typename T>
mlk::GemmTestCase<T>::GemmTestCase(
    mlk::Matrix<T> left,
    mlk::Matrix<T> right,
    mlk::Matrix<T> output,
    std::string name
) : left_(left), right_(right), output_(output), name_(name) {}

template<typename T>
mlk::GemmTestCase<T>::GemmTestCase(std::size_t M,
               std::size_t K,
               std::size_t N,
               std::mt19937_64& gen,
               std::string name
) : left_(M, K, gen), right_ (K, N, gen), output_(M, N), name_(name) {

    reordered_gemm<T>(left_, right_ , output_);
}

template<typename T>
mlk::GemmTestCase<T>::GemmTestCase(std::size_t M,
             std::size_t K,
             std::size_t N,
             std::mt19937_64& gen)
: GemmTestCase(M, K, N, gen, "Generic_Random") {}

template<typename T>
const std::string& mlk::GemmTestCase<T>::name() const { return name_; }

template<typename T>
const mlk::Matrix<T>& mlk::GemmTestCase<T>::left() const { return left_; }

template<typename T>
const mlk::Matrix<T>& mlk::GemmTestCase<T>::right() const {return right_; }

template<typename T>
const mlk::Matrix<T>& mlk::GemmTestCase<T>::output() const {return output_; }

template<typename T>
void mlk::GemmTestCase<T>::write(std::ostream& output) const {

    output << "Left ";
    left_.write(output);
    output << "Right ";
    right_.write(output);
    output << "Output ";
    output_.write(output);
}

template<typename T>
void mlk::GemmTestCase<T>::randomize(std::mt19937_64& gen) {

    left_.randomize(gen);
    right_.randomize(gen);
    output_.randomize(gen);
}

template
class mlk::Matrix<float>;

template
class mlk::Matrix<double>;

template
class mlk::GemmTestCase<float>;

template
class mlk::GemmTestCase<double>;
