// Copyright 2023 Amit Goren

#include <anmlriddle/server/inference.h>

#include <ranges>
#include <numeric>

#include <anmlriddle/com.h>
#include <anmlriddle/unexpected_message_error.h>
#include <thread>
#include "../infer_layer.h"
#include "Eigen/src/Core/Matrix.h"
#include "anmlriddle/channel.h"
#include "../client_message_generated.h"
#include "split_model.h"

namespace anmlriddle {

namespace server {

// XXX XXX FINAL LAYER??????
// NOTE idea: seperate to more classes (e.g. LayerInference class)
  
void Inference::Infer(std::span<const std::byte> model_buffer) {
  #warning this function should be guarded

  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(model_buffer.data()), model_buffer.size_bytes());
  if (!VerifyModelBuffer(verifier)) {
    throw std::runtime_error(
        "Inference::Inference: tried to read an invalid model");
  }
      
  const Model* model = GetModel(static_cast<const void*>(model_buffer.data()));
  SplitModel(model, std::make_pair(&model_share_, &client_model_share_buf_));  // NOTE should I add & before model_share_?

  std::jthread infer_layers(
    [&] (std::stop_token stop_token) { InferLayers(stop_token); });
  
  // Throw an exception if one occurs
  result_.get_future().get();
}

void Inference::InferLayers(std::stop_token stop_token) noexcept {    
  try {
    // Send the client its model share
    send_(client_model_share_buf_);

    // Wait for the input share
    const Dense* input_share = input_share_.Read(stop_token);
    if (!input_share) {
      return;
    }

    Eigen::VectorX<Com> activations_share = AsEigenVector(input_share);

    // Evaluate each layer
    for (auto layer_share : model_share_.layerShares) {
      activations_share = InferLayer(layer_share, activations_share, &GetMT, stop_token,
                                     mt_inference_share_, send_);
    }

    #warning send the client the output share

    result_.set_value();
  } catch (...) {
    result_.set_exception(std::current_exception());
  }
}

void Inference::Receive(std::vector<std::byte> message_buffer) {
  // Verify the message is valid
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(message_buffer.data()), message_buffer.size());
  if (!VerifyClientMessageBuffer(verifier)) {
    throw UnexpectedMessageError(
        "Inference::Receive: tried to read a corrupt message");
  }

  // Parse the message
  const ClientMessage* message = GetClientMessage(
      static_cast<const void*>(message_buffer.data()));
  
  // Multiplex the message
  switch (message->message_type()) {
   case ClientMessageUnion_InputShare:
    input_share_.Insert(message_buffer, message->message_as_InputShare());
    break;
   case ClientMessageUnion_MTInferenceShare:
    mt_inference_share_.Insert(message_buffer, message->message_as_MTInferenceShare());
    break;
   default:
    throw UnexpectedMessageError("Inference::Receive: received a message of unknown type");
  }
}

}  // namespace server

}  // namespace anmlriddle
