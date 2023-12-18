// Copyright 2023 Amit Goren

// Utilities and aliases for the 'communicable' type, a fixed-point number to
// which all numbers are encoded for cryptographic purposes.

#ifndef ANMLRIDDLE_COM_H_
#define ANMLRIDDLE_COM_H_

#include <cstdint>
#include <vector>

#include <capnp/list.h>

// The size of the fraction in bits
const int kFractionBits = 4;

// The communicable type for cryptographic use. All the numbers (even floats)
// are converted to it. 16 bits corresponds to Z_16.
using Com = uint16_t;

using ComList = ::capnp::List<Com>;

using ComVec = std::vector<Com>;

inline Com FloatToCom(float x) {
  return x * (1 << kFractionBits);
}

inline float ComToFloat(Com x) {
  float y = x;
  return y / (1 << kFractionBits);
}

#endif  // ANMLRIDDLE_COM_H_
