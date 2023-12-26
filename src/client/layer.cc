// Copyright 2023 Amit Goren

#include <anmlriddle/client/layer.h>

namespace amrc {

void Layer::Send(const ::capnp::MessageBuilder& message) {
  inference_.send_(message);
}

ServerMessage::Reader Layer::GetMessage() {
  std::unique_lock lock(inference_.layer_messages_mutex_);

  if (inference_.layer_messages_.empty()) {
    inference_.layer_messages_condition_.wait(lock);
  }

  auto message = inference_.layer_messages_.front();
  inference_.layer_messages_.pop();
  return message;
}

}  // namespace amrc
