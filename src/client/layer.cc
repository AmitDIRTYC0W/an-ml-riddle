// Copyright 2023 Amit Goren

#include <anmlriddle/client/layer.h>

namespace amrc {

void Layer::Send(const ::capnp::MessageBuilder& message) {
  inference_.send_(message);
}

::capnp::MessageReader& GetMessage() {
  std::unique_lock lock(inference_.layer_messages_mutex_);

  if (inference_.layer_messages_.empty()) {
    inference_.layer_messages_condition_.wait(lock)
  }
  return inference_.layer_messages_.pop();
}

}  // namespace amrc
