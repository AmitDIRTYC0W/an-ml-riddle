// Copyright 2023 Amit Goren

#include <anmlriddle/client/inference.h>

#include <algorithm>
#include <execution>

#include "../generate_shares.h"
#include <anmlriddle/client/initial_layer.h>
#include <anmlriddle/client/final_layer.h>
#include <anmlriddle/client/dense_layer.h>
#include <anmlriddle/com.h>
#include "Model.capnp.h"

namespace amrc {

void Inference::SetupModel(const ModelShare::Reader& model_share) {
  layers_.push_back(std::make_unique<InitialLayer>(*this));
    
  auto layer_shares = model_share.getLayerShares();
  for (auto it = layer_shares.begin(); it != layer_shares.end(); ++it) {
    switch (it->which()) {
      case LayerShare::DENSE:
        layers_.push_back(std::make_unique<DenseLayer>(*this, it->getDense()));
        break;
    }
  }

  layers_.push_back(std::make_unique<FinalLayer>(*this));
}

void Inference::StartOnlinePhase() {
  Inference::layers_inference_ = std::jthread(&Inference::InferLayers, this);
}

void Inference::InferLayers() {
  using namespace std::placeholders;

  // Convert the input from float to communicable
  ComVec input_com(input_.size());
  std::transform(std::execution::par_unseq, input_.begin(), input_.end(),
      input_com.begin(), FloatToCom);

  // Iterate over all the layers and execute each one
  try {
    ComVec output_com = std::accumulate(layers_.begin(), layers_.end(),
        input_com, std::bind(&Layer::Infer, _2, _1));

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

void Inference::Receive(::capnp::MessageReader& message) {
  auto server_message = message.getRoot<ServerMessage>();

  switch (server_message.which()) {
    case ServerMessage::MODEL_SHARE:
      if (!received_model_) {
        received_model_ = true;
        SetupModel(server_message.getModelShare());
        StartOnlinePhase();
      } else {
        result_.set_exception(
          std::make_exception_ptr(
            new UnexpectedMessageError("Received another model share")));
      }
      break;
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
