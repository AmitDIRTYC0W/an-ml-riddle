// Copyright 2024 Amit Goren

#include <anmlriddle/unexpected_message_error.h>
#include "dense_layer.h"
#include "anmlriddle/channel.h"
#include "common_generated.h"
#include "split_dense.h"
#include <anmlriddle/multiplication_triplet.h>
#include <anmlriddle/com.h>
#include <stop_token>

namespace anmlriddle {

unsigned short DenseLayerOutputSize(const DenseLayerShareT& share) {
  return share.biasesShare.size();
}

unsigned short DenseLayerOutputSize(const DenseLayerShare* share) {
  return share->biasesShare()->size();
}

// TODO combine these two functions commonities
void InferDenseLayer(const DenseLayerShareT& share,
                     const Eigen::Ref<const Eigen::MatrixX<Com>> input,
                     Eigen::Ref<Eigen::MatrixX<Com>> output, MTProvider get_mt,
                     SingularSink<MTInferenceShare>& messagesSink, SendFunction send,
                     std::stop_token stop_token) {
  MultiplicationTriplet mt = get_mt(1, input.size(), share.biasesShare.size());
  auto matmul_share = mt.Multiply(input, AsEigenMatrix(*share.weightsShare), send, messagesSink, stop_token);
  auto biases_share = Eigen::Map<const Eigen::VectorX<Com>>(share.biasesShare.data(), share.biasesShare.size()); 
  output = (biases_share + matmul_share).eval();
}

void InferDenseLayer(const DenseLayerShare* share,
                     const Eigen::Ref<const Eigen::MatrixX<Com>> input,
                     Eigen::Ref<Eigen::MatrixX<Com>> output, MTProvider get_mt,
                     SingularSink<MTInferenceShare>& messagesSink, SendFunction send,
                     std::stop_token stop_token) {
  MultiplicationTriplet mt = get_mt(1, input.size(), share->biasesShare()->size());
  auto matmul_share = mt.Multiply(input, AsEigenMatrix(share->weightsShare()), send, messagesSink, stop_token);
  auto biases_share = Eigen::Map<const Eigen::VectorX<Com>>(share->biasesShare()->data(), share->biasesShare()->size()); 
  output = (biases_share + matmul_share).eval();
}

LayerSharePair SplitDenseLayer(const DenseLayer* layer, flatbuffers::FlatBufferBuilder& builder) {
  std::size_t output_size = layer->biases()->size();
  std::size_t weights_size = layer->weights()->values()->size();
  #warning add assertions wherever output_size is used
 
  // WARNING This must be deleted manually
  DenseLayerShareT* first_share = new DenseLayerShareT();
  
  // Split the biases into shares
  first_share->biasesShare = std::vector<Com>(output_size);
  auto [second_biases_share_flatbuffers, second_biases_share_eigen] = FlatbuffersVector(output_size, builder);
  Eigen::Map<const Eigen::VectorX<Com>> secret_biases_eigen(layer->biases()->data(), output_size);
  Eigen::Map<Eigen::VectorX<Com>> first_biases_share_eigen(first_share->biasesShare.data(), output_size);
  SplitDense(secret_biases_eigen, first_biases_share_eigen, second_biases_share_eigen);

  // Split the weights into share
  first_share->weightsShare = std::make_unique<MatrixT>();
  first_share->weightsShare->values = std::vector<Com>(weights_size);
  unsigned short weights_rows = RowsInMatrix(weights_size, layer->weights()->columns());
  auto [second_weights_share_flatbuffers, second_weights_share_eigen] = FlatbuffersMatrix(
      weights_rows, layer->weights()->columns(), builder);
  auto secret_weights_eigen = AsEigenMatrix(layer->weights());
  auto first_weights_share_eigen = AsEigenMatrix(*(first_share->weightsShare));
  SplitDense(secret_weights_eigen, first_weights_share_eigen, second_weights_share_eigen);
  
  auto second_share = CreateDenseLayerShare(builder, second_weights_share_flatbuffers,
                                            second_biases_share_flatbuffers);
  
  LayerShareUnion first_share_union;
  first_share_union.type = LayerShare_DenseLayerShare;
  first_share_union.value = first_share;
  
  return std::make_pair(first_share_union, second_share.Union());
}

}  // namespace anmlriddle