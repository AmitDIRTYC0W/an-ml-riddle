// Copyright 2024 Amit Goren

#include <anmlriddle/unexpected_message_error.h>
#include "dense_layer.h"
#include <anmlriddle/multiplication_triplet.h>
#include <anmlriddle/com.h>

namespace amr {

unsigned short DenseLayerOutputSize(const DenseLayerShareT& share) {
  return share.biasesShare.size();
}

void InferDenseLayer(const DenseLayerShareT& share,
                     const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                     Eigen::Ref<Eigen::MatrixX<Com>> output, MTProvider get_mt,
                     IO<> io) {
  MultiplicationTriplet mt = get_mt(1, input.size(), share.biasesShare.size());
  auto matmul_share = mt.Multiply(input, AsComMatrix(*share.weightsShare), io);
  output = (Eigen::Map<const Eigen::VectorX<Com>>(share.biasesShare.data(), share.biasesShare.size()) + matmul_share).eval();
}

}  // namespace amr