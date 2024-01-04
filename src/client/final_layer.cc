// Copyright 2023 Amit Goren

#include <anmlriddle/client/final_layer.h>

#include <algorithm>
#include <execution>

#include <anmlriddle/client/unexpected_message_error.h>
#include <anmlriddle/com.h>
#include "Model.capnp.h"

namespace amrc {

ComVec FinalLayer::Infer(ComVec last) {
  // The server should send us a message
  auto servers_share = FetchMessage().getVectorShare().getVectorShare();

  if (last.size() != servers_share.size()) {
    throw UnexpectedMessageError("The server's share and ours differ in size");
  }

  ComVec result(last.size());
    // FIXME use par_unseq
  std::transform(servers_share.begin(), servers_share.end(),
      last.begin(), result.begin(), std::plus<Com>());

  return result;
}

}  // namespace amrc
