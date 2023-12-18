// Copyright 2023 Amit Goren

#include <anmlriddle/client/dense_layer.h>

#include <algorithm>
#include <functional>

#include <anmlriddle/com.h>

namespace amrc {

void DenseLayer::Begin() {
  auto biases_share = layer_share_.getBiasesShare();

  /*
  if (inference_.result_share_.size() != biases_share.size()) {
    throw UnexpectedMessageError("The server's share and ours differ in size");
  }*/

  auto new_result_share = std::make_shared<ComVec>();
  std::transform(get_result_share_().begin(), get_result_share_().end(),
      biases_share.begin(), new_result_share->begin(), std::plus<Com>());
  set_result_share_(new_result_share);

  Next();
}

}  // namespace amrc
