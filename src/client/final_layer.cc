// Copyright 2023 Amit Goren

#include <anmlriddle/client/final_layer.h>

#include <algorithm>

#include <anmlriddle/client/unexpected_message_error.h>
#include <anmlriddle/com.h>
// #include "vector_share_generated.h"
#include "Model.capnp.h"

namespace amrc {

inline float SumComToFloat(const Com& a, const Com& b) {
  return ComToFloat(a + b);
}

void FinalLayer::Receive(::capnp::MessageReader& message) {
  auto servers_share = message.getRoot<VectorShare>().getVectorShare();

  if (get_result_share_().size() != servers_share.size()) {
    throw UnexpectedMessageError("The server's share and ours differ in size");
  }

  std::vector<float> result;
  std::transform(get_result_share_().begin(),
      get_result_share_().end(), servers_share.begin(), result.begin(),
      SumComToFloat);

  get_result_().set_value(result);
}

}  // namespace amrc
