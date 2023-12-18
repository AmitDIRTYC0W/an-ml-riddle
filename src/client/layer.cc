// Copyright 2023 Amit Goren

#include <anmlriddle/client/layer.h>

namespace amrc {

void Layer::set_model_share_(ModelShare::Reader model_share) {
  inference_.set_model_share_(model_share);
}

ComVec& Layer::get_result_share_() {
  return *inference_.result_share_;
}

void Layer::set_result_share_(std::shared_ptr<ComVec> result_share) {
  inference_.result_share_ = result_share;
}

void Layer::Next() {
  inference_.Next();
}

std::promise<std::vector<float>>& Layer::get_result_() {
  return inference_.result_;
}

void Layer::Send(const ::capnp::MessageBuilder& message) {
  inference_.send_(message);
}

}  // namespace amrc
