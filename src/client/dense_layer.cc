// Copyright 2023 Amit Goren

#include <anmlriddle/client/dense_layer.h>

#include <algorithm>
#include <functional>

#include <anmlriddle/com.h>
#include <anmlriddle/client/unexpected_message_error.h>

namespace amrc {

ComVec DenseLayer::Infer(ComVec last) {
  auto biases_share = layer_share_.getBiasesShare();
  
  if (last.size() != biases_share.size()) {
    throw UnexpectedMessageError("The server's share and ours differ in size");
  }

  ComVec sum_share(last.size());
  std::transform(last.begin(), last.end(), biases_share.begin(),
      sum_share.begin(), std::plus<Com>());

  return sum_share;
}

}  // namespace amrc
