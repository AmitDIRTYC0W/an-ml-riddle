// Copyright 2023 Amit Goren

#include <anmlriddle/client/inference.h>

#include <algorithm>
#include <execution>
#include <ranges>

#include "../generate_shares.h"
#include <anmlriddle/client/final_layer.h>
#include <anmlriddle/client/dense_layer.h>
#include <anmlriddle/com.h>
#include "Model.capnp.h"

namespace amrc {

ServerMessage::Reader Inference::FetchMessage() {
  std::unique_lock lock(layer_messages_mutex_);

  if (layer_messages_.empty()) {
    layer_messages_condition_.wait(lock);
  }

  auto message = layer_messages_.front();
  layer_messages_.pop();
  return message;
}

ComVec Inference::SendInputShare(ComVec input) {
  // Split the input into shares
  ::capnp::MallocMessageBuilder message_to_server;
  auto servers_share = message_to_server.initRoot<VectorShare>();
  auto servers_input_share = servers_share.initVectorShare(input.size());

  // Generate both shares
  ComVec our_input_share(input.size());
  GenerateShares(input,
      std::pair<ComVec&, ComList::Builder&>(our_input_share,
        servers_input_share));

  // Send a share to the server
  send_(message_to_server);

  return our_input_share;
}

auto Inference::FetchLayers() {
  // Wait for the model share
  auto layer_shares = FetchMessage().getModelShare().getLayerShares();

  // Add the layers to the inference
  std::vector<std::unique_ptr<Layer>> layers(layer_shares.size());
  std::transform(layer_shares.begin(), layer_shares.end(), layers.begin(),
      [&](LayerShare::Reader share) -> std::unique_ptr<Layer> {
        switch (share.which()) {
          case LayerShare::DENSE:
            return std::make_unique<DenseLayer>(*this, share.getDense());
          default:
            return nullptr; // FIXME raise a new type of exception
        }
      });
  layers.push_back(std::make_unique<FinalLayer>(*this));

  return layers;
}

void Inference::InferLayers() noexcept {
  using namespace std::placeholders;

  // Convert the input from float to communicable
  ComVec input_com(input_.size());
  std::transform(std::execution::par_unseq, input_.begin(), input_.end(),
      input_com.begin(), FloatToCom);

  // Iterate over all the layers and execute each one
  try {
    ComVec our_input_share = SendInputShare(input_com);
    auto layers = FetchLayers();
    ComVec output_com = std::accumulate(layers.begin(), layers.end(), our_input_share,
      std::bind(&Layer::Infer, _2, _1));

    // Convert the output back to float
    std::vector<float> output(output_com.size());
    std::transform(output_com.begin(), output_com.end(), output.begin(),
        ComToFloat);
    result_.set_value(output);
  } catch (...) {
    // WARNING I'm not sure std::accumulate propogates exceptions...
    result_.set_exception(std::current_exception());
  }
}

void Inference::Begin() {
  layers_inference_ = std::jthread(&Inference::InferLayers, this);
}

void Inference::Receive(::capnp::MessageReader& message) noexcept {
  auto server_message = message.getRoot<ServerMessage>();

  switch (server_message.which()) {
    case ServerMessage::MODEL_SHARE:
    case ServerMessage::VECTOR_SHARE: // WARNING This will be changed
      {
        std::scoped_lock<std::mutex> layer_messages_lock(layer_messages_mutex_);
        layer_messages_.push(server_message);
      }
      layer_messages_condition_.notify_one();
      break;
  }
}

}  // namespace amrc
