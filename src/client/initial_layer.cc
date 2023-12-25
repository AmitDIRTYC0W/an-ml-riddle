// Copyright 2023 Amit Goren

#include <anmlriddle/client/initial_layer.h>

// #include <amrc/Model.capnp.h>
#include "Model.capnp.h"
#include <anmlriddle/com.h>
#include "../generate_shares.h"

namespace amrc {

ComVec InitialLayer::Infer(ComVec last) {
  // Split the input into shares
  ::capnp::MallocMessageBuilder message_to_server;
  auto servers_share = message_to_server.initRoot<VectorShare>();
  auto servers_input_share = servers_share.initVectorShare(last.size());

  // Generate both shares
  ComVec our_input_share(last.size());
  GenerateShares(last,
      std::pair<ComVec&, ComList::Builder&>(servers_input_share,
        our_input_share));

  // Send a share to the server
  Send(message_to_server);

  return our_input_share;
}

}  // namespace amrc
