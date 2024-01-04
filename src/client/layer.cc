// Copyright 2023 Amit Goren

#include <anmlriddle/client/layer.h>

namespace amrc {

void Layer::Send(const ::capnp::MessageBuilder& message) {
  inference_.send_(message);
}

ServerMessage::Reader Layer::FetchMessage() {
  return inference_.FetchMessage();
}

}  // namespace amrc
