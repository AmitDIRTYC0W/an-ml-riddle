// Copyright 2023 Amit Goren

#include <anmlriddle/client/inference.h>

#include <ranges>
#include <stop_token>
#include <thread>

#include "../split_dense.h"
#include <anmlriddle/unexpected_message_error.h>
#include <anmlriddle/com.h>
#include "../client_message_generated.h"
#include "../infer_layer.h"

namespace anmlriddle {
// 
namespace client {

inline std::pair<Eigen::VectorX<Com>, flatbuffers::DetachedBuffer> SplitInput(std::span<const float> input) {
  // Convert the input to Com
  auto input_com = FloatToCom(input);
  
  // Prepare the first share
  Eigen::VectorX<Com> first_share(input.size());
  
  // Prepare the second share inside a flatbuffers builder
  flatbuffers::FlatBufferBuilder second_share_builder;
  auto [second_share_flatbuffers, second_share_eigen] = FlatbuffersDense(input.size(), second_share_builder);
  
  // Split the shares
  SplitDense(input_com, first_share, second_share_eigen);
  
  // Finish the buffer of the second share
  auto message = CreateClientMessage(second_share_builder, ClientMessageUnion_InputShare,
                                     second_share_flatbuffers.Union());
  second_share_builder.Finish(message);
  
  return {first_share, second_share_builder.Release()};
}

// TODO lose the std::pair
inline std::vector<float> ReconstructOutput(std::pair<Eigen::VectorX<Com>, const Dense*> shares) {
  #warning need to assert shares.first.size() == shares.second->???->size();
  
  auto second_share_flatbuffers = shares.second->values();
  Eigen::Map<const Eigen::VectorX<Com>> second_share_eigen(second_share_flatbuffers->data(),
                                                     second_share_flatbuffers->size());
  
  std::vector<Com> output_com(shares.first.size());
  Eigen::Map<Eigen::VectorX<Com>> output_eigen(output_com.data(), output_com.size());
  
  output_eigen = shares.first + second_share_eigen;
  
  return ComToFloat(output_com);
}

std::vector<float> Inference::Infer(std::span<float> input) {
  #warning this function should be guarded like so:
  /*
    if (began_) {
    throw std::runtime_error("Inference::Begin may be called only once");
  }
  began_ = true;
  */

  
  // TODO split the input here, not in the seperate thread
  std::jthread infer_layers(
      [&] (std::stop_token stop_token, std::span<float> input) { InferLayers(stop_token, input); },
      input);
  return output_promise_.get_future().get();
}

void Inference::Receive(std::vector<std::byte>& message_buffer) {
  // Verify the message is valid
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(message_buffer.data()), message_buffer.size());
  if (!VerifyServerMessageBuffer(verifier)) {
      throw UnexpectedMessageError("Inference::Receive: received an invalid message");
  }
  
  // Parse the message
  const ServerMessage* message = GetServerMessage(static_cast<void*>(message_buffer.data()));
  
  // Multiplex the message
  switch (message->message_type()) {
   case ServerMessageUnion_ModelShare:
    model_share_.Insert(message_buffer, message->message_as_ModelShare());
    break;
   case ServerMessageUnion_MTInferenceShare:
    mt_inference_share_.Insert(message_buffer, message->message_as_MTInferenceShare());
    break;
   case ServerMessageUnion_OutputShare:
    their_output_share_.Insert(message_buffer, message->message_as_OutputShare());
    break;
   default:
    throw UnexpectedMessageError("Inference::Receive: received a message of an unknown type");
  }
}

void Inference::InferLayers(std::stop_token stop_token, std::span<float> input) noexcept {
  try {
    // Split the input into shares and send the server its share
    auto [activations_share, their_input_share] = SplitInput(input);
    send_(their_input_share);
  
    // Wait for the model share
    const ModelShare* model_share = model_share_.Read(stop_token);
    if (!model_share) {
      return;
    }
    
    // Evaluate each layer
    auto layer_share = model_share->layerShares()->cbegin();
    auto layer_type = model_share->layerShares_type()->cbegin();
    while (layer_share != model_share->layerShares()->cend()) {
      activations_share = InferLayer(static_cast<void*>(&layer_share), static_cast<const LayerShare>(*layer_type),
                               activations_share, &GetMT, stop_token, mt_inference_share_, send_);

      ++layer_share;
      ++layer_type;
    }
    
    // Wait for the server's output share and reconstruct the outupt
    output_promise_.set_value(ReconstructOutput(std::make_pair(activations_share, their_output_share_.Read())));
  } catch (...) {
    output_promise_.set_exception(std::current_exception());
  }
}

}  // namespace client

}  // namespace anmlriddle
