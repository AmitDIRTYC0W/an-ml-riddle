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

#include "common_generated.h"

namespace amr {

// The size of the fraction in bits
const int kFractionBits = 4;

// The communicable type for cryptographic use. All the numbers (even floats)
// are converted to it. 16 bits correspond to Z_16.
using Com = int16_t;

inline Com FloatToCom(float x) noexcept {
  return x * (1 << kFractionBits);
}

inline float ComToFloat(Com x) noexcept {
  float y = x;
  return y / (1 << kFractionBits);
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

inline Eigen::Map<Eigen::MatrixX<Com>> AsComMatrix(
    std::span<Com> values, unsigned short columns) {
  unsigned short rows = RowsInMatrix(values.size(), columns);
  return Eigen::Map<Eigen::MatrixX<Com>>(values.data(), rows, columns);
}

inline Eigen::Map<const Eigen::MatrixX<Com>> AsConstComMatrix(
    std::span<const Com> values, unsigned short columns) {
  unsigned short rows = RowsInMatrix(values.size(), columns);
  return Eigen::Map<const Eigen::MatrixX<Com>>(values.data(), rows, columns);
}

inline Eigen::Map<const Eigen::MatrixX<Com>> AsComMatrix(const Matrix* matrix) {
  std::span values = {matrix->values()->data(), matrix->values()->size()};
  return AsConstComMatrix(values, matrix->columns());
}

/*inline Eigen::Map<const Eigen::MatrixX<Com>> AsComMatrix(const MatrixT matrix) {
  std::span values = {matrix.values.data(), matrix.values.size()};
  return AsComMatrix(values, matrix.columns);
}*/

inline Eigen::Map<Eigen::MatrixX<Com>> AsComMatrix(MatrixT matrix) {
  std::span values = {matrix.values.data(), matrix.values.size()};
  return AsComMatrix(values, matrix.columns);
}

inline auto FlatbuffersComMatrix(Eigen::Index rows, Eigen::Index columns,
                                 flatbuffers::FlatBufferBuilder& builder) {
  Com* buffer;
  auto values = builder.CreateUninitializedVector<Com>(rows * columns, &buffer);
  auto flatbuffers_matrix = CreateMatrix(builder, values, columns);
  Eigen::Map<Eigen::MatrixX<Com>> eigen_matrix(buffer, rows, columns); // TODO eigen_matrix -> matrix_eigen and likewise
  return std::pair(flatbuffers_matrix, eigen_matrix);
}

inline auto FlatbuffersComVec(Eigen::Index size,
                              flatbuffers::FlatBufferBuilder& builder) {
  Com* buffer;
  auto values = builder.CreateUninitializedVector<Com>(size, &buffer);
  auto vector_flatbuffers = CreateVector(builder, values);
  Eigen::Map<Eigen::VectorX<Com>> vector_eigen(buffer, size);
  return std::pair(vector_flatbuffers, vector_eigen);
}

inline auto FlatbuffersComVector(Eigen::Index size,
                                 flatbuffers::FlatBufferBuilder& builder) {
  Com* buffer;
  auto values = builder.CreateUninitializedVector<Com>(size, &buffer);
  Eigen::Map<Eigen::VectorX<Com>> vector_eigen(buffer, size);
  return std::pair(values, vector_eigen);
}

// inline std::span<Com> AsSpan(capnp::Data::Reader& data) {
//   if (data.size() % sizeof(Com) != 0) {
//     throw std::invalid_argument("AsSpan: data buffer might not depict Coms");
//   }

//   return {data.asArray().begin(), data.size() / sizeof(Com)};
// }

}  // namespace amr

#endif  // ANMLRIDDLE_COM_H_
