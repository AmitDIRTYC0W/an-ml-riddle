// Copyright 2023 Amit Goren

#include <anmlriddle/server/inference.h>

#include <ranges>
#include <numeric>

#include <anmlriddle/com.h>
#include <anmlriddle/unexpected_message_error.h>
#include "../infer_layer.h"
#include "client_message_generated.h"
#include "split_model.h"

namespace anmlriddle {

namespace server {

// XXX XXX FINAL LAYER??????
// NOTE idea: seperate to more classes (e.g. LayerInference class)
  
Inference::Inference(
    std::function<void(const flatbuffers::DetachedBuffer&)> send,
    std::span<const std::byte> model_buf) : send_(send),
                                            layer_messages_(std::make_shared<SynchronisedQueue<>>()),
                                            layer_io_(IO<>(layer_messages_, send)) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(model_buf.data()), model_buf.size_bytes());
  if (!VerifyModelBuffer(verifier)) {
    throw std::runtime_error(
        "Inference::Inference: tried to read an invalid model");
  }
      
  const Model* model = GetModel(static_cast<const void*>(model_buf.data()));
  SplitModel(model, std::make_pair(&model_share_, &client_model_share_buf_)); // NOTE should I add & before model_share_?
}

std::future<void> Inference::Begin() {
  if (began_) {
    throw std::runtime_error("Inference::Begin may be called only once");
  }
  began_ = true;

  send_(client_model_share_buf_);

  layers_inference_ = std::jthread(&Inference::InferLayers, this);
  
  return result_.get_future();
}

void Inference::InferLayers() noexcept {    
  try {
    // Wait for the input share
    std::unique_lock input_lock(input_mutex_);
    input_condition_.wait(input_lock,
                          std::bind(&decltype(input_)::has_value, input_));
    auto input_message = GetClientMessage(input_->data());
    auto input_values = input_message->message_as_InputShare()->values();
    Eigen::Map<const Eigen::VectorX<Com>> input_vector(input_values->data(), input_values->size());

    Eigen::VectorX<Com> activations = input_vector;
    for (auto layer : model_share_.layerShares) {
      activations = InferLayer(layer, activations, &GetMT, layer_io_);
    }
    
    #warning Actually send the value to the client
    /* TODO XXX This shall no be done here (ultimately), this function shall only 
     * return the output_share  (or, alternatively, move the sending here too)*/

    result_.set_value();
  } catch (...) {
    result_.set_exception(std::current_exception());
  }
}

void Inference::Receive(const std::vector<std::byte> message) {
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(message.data()), message.size());
  if (!VerifyClientMessageBuffer(verifier)) {
    throw UnexpectedMessageError(
        "Inference::Receive: tried to read a corrupt message");
  }
  const ClientMessage* client_message = GetClientMessage(
      static_cast<const void*>(message.data()));
  
  switch (client_message->message_type()) {
   case ClientMessageUnion_InputShare:
    {
      std::scoped_lock<std::mutex> input_lock(input_mutex_);
      input_ = message;
    }
    input_condition_.notify_all();
    break;
   case ClientMessageUnion_MTInferenceShare:
    layer_messages_->Push(message);
    break;
   default:
    throw UnexpectedMessageError("Inference::Receive: received a message of unknown type");
  }
}

}  // namespace server

}  // namespace anmlriddle
