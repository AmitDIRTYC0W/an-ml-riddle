// Copyright 2023 Amit Goren

// Utilities and aliases for the 'communicable' type, a fixed-point number to
// which all numbers are encoded for cryptographic purposes.

#ifndef ANMLRIDDLE_COM_H_
#define ANMLRIDDLE_COM_H_

#include <cstdint>
#include <span>
#include <stdexcept>

#include <Eigen/Dense>
#include <flatbuffers/flatbuffers.h>

// #include "common_generated.h"
#include "../../src/common_generated.h"
// #include "common_generated.h"

namespace anmlriddle {

// The size of the fraction in bits
const int kFractionBits = 4;

// The communicable type for cryptographic use. All the numbers (even floats)
// are converted to it. 16 bits correspond to Z_16.
using Com = int16_t;

inline Com FloatToCom(float x) noexcept {
  return x * (1 << kFractionBits);
}

inline auto FloatToCom(std::span<const float> x) noexcept {
  const Eigen::Map<const Eigen::VectorXf> x_eigen(x.data(), x.size());
  return (x_eigen * (1 << kFractionBits)).cast<Com>();
}

inline float ComToFloat(Com x) noexcept {
  float y = x;
  return y / (1 << kFractionBits);
}

inline std::vector<float> ComToFloat(std::span<const Com> x) noexcept {
  std::vector<float> output_std(x.size());
  Eigen::Map<Eigen::VectorXf> output_eigen(output_std.data(), output_std.size());
  const Eigen::Map<const Eigen::VectorX<Com>> x_eigen(x.data(), x.size());
  output_eigen = x_eigen.cast<float>() / (1 << kFractionBits);
  return output_std;
}

template <typename T>
concept ComRange = std::ranges::range<T>
                   && std::same_as<std::ranges::range_value_t<T>, Com>;

// Call this after multiplying two Coms to account for the implicit
// multiplication of (1 << kFractionBits) by itself.
template <typename T> // TODO add constraint
inline auto AdjustMultiplication(T x) {
  return x / (1 << kFractionBits);
}

inline unsigned short RowsInMatrix(unsigned short size, unsigned short columns) {
    if (size % columns != 0) {
        throw std::runtime_error("RowsInMatrix: the buffer cannot fit the alleged amount of columns");
    }
    
    return size / columns;
}

inline Eigen::Map<Eigen::MatrixX<Com>> AsEigenMatrix(
    std::span<Com> values, unsigned short columns) {
  unsigned short rows = RowsInMatrix(values.size(), columns);
  return Eigen::Map<Eigen::MatrixX<Com>>(values.data(), rows, columns);
}

inline Eigen::Map<const Eigen::MatrixX<Com>> AsEigenMatrix(
    std::span<const Com> values, unsigned short columns) {
  unsigned short rows = RowsInMatrix(values.size(), columns);
  return Eigen::Map<const Eigen::MatrixX<Com>>(values.data(), rows, columns);
}

inline Eigen::Map<const Eigen::MatrixX<Com>> AsEigenMatrix(const Matrix* matrix) {
  std::span values = {matrix->values()->data(), matrix->values()->size()};
  return AsEigenMatrix(values, matrix->columns());
}

inline Eigen::Map<Eigen::MatrixX<Com>> AsEigenMatrix(MatrixT matrix) {
  std::span values = {matrix.values.data(), matrix.values.size()};
  return AsEigenMatrix(values, matrix.columns);
}

inline auto FlatbuffersMatrix(Eigen::Index rows, Eigen::Index columns,
                                 flatbuffers::FlatBufferBuilder& builder) {
  Com* buffer;
  auto values = builder.CreateUninitializedVector<Com>(rows * columns, &buffer);
  auto flatbuffers_matrix = CreateMatrix(builder, values, columns);
  Eigen::Map<Eigen::MatrixX<Com>> matrix_eigen(buffer, rows, columns);
  return std::pair(flatbuffers_matrix, matrix_eigen);
}

inline auto FlatbuffersDense(Eigen::Index size,
                              flatbuffers::FlatBufferBuilder& builder) {
  Com* buffer;
  auto values = builder.CreateUninitializedVector<Com>(size, &buffer);
  auto dense = CreateDense(builder, values);
  Eigen::Map<Eigen::VectorX<Com>> vector_eigen(buffer, size);
  return std::pair(dense, vector_eigen);
}

inline auto FlatbuffersVector(Eigen::Index size,
                              flatbuffers::FlatBufferBuilder& builder) {
  Com* buffer;
  auto vector_flabuffers = builder.CreateUninitializedVector<Com>(size, &buffer);
  Eigen::Map<Eigen::VectorX<Com>> vector_eigen(buffer, size);
  return std::pair(vector_flabuffers, vector_eigen);
}

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_COM_H_
