// Copyright 2023 Amit Goren

#include <anmlriddle/client/initial_layer.h>

// #include <amrc/Model.capnp.h>
#include "Model.capnp.h"
#include <anmlriddle/com.h>
#include "../generate_shares.h"

namespace amrc {

void InitialLayer::Receive(::capnp::MessageReader& message) {
  // The first message must be a ModelShare. We shall store it.
  set_model_share_(message.getRoot<ModelShare>());

  // Then, we split the input into shares
  ::capnp::MallocMessageBuilder message_to_server;
  auto servers_share = message_to_server.initRoot<VectorShare>();
  auto servers_input_share = servers_share.initVectorShare(input_.size());

  // Convert the input from float to communicable
  ComVec input_com = ComVec(input_.size());
  for (size_t i = 0; i < input_.size(); ++i) {
    input_com[i] = FloatToCom(input_[i]);
  }

  // Generate both shares
  GenerateShares(input_com,
      std::pair<ComVec&, ComList::Builder&>(get_result_share_(),
        servers_input_share));

  // Send to the server its share
  Send(message_to_server);

  // Move to the first layer
  Next();
}

}  // namespace amrc
