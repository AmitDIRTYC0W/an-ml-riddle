// Copyright 2023 Amit Goren

#include <anmlriddle/client/inference.h>

#include "../generate_shares.h"
#include <anmlriddle/client/initial_layer.h>
#include <anmlriddle/client/final_layer.h>
#include <anmlriddle/client/dense_layer.h>
#include <anmlriddle/com.h>

namespace amrc {

void Inference::SetupModel(const ModelShare::Reader& model_share) {
  layers_->push_back(InitialLayer());
    
  auto layer_shares = model_share.getLayerShares();
  for (auto it = layer_shares.begin(); it != layer_shares.end(); ++it) {
    switch (it->which()) {
      case LayerShare::DENSE:
        layers_->push_back(DenseLayer(it->getDense()));
        break;
    }
  }

  layers_->push_back(FinalLayer());
}

void Inference::StartOnlinePhase() {
  layers_inference_ = std::jthread(InferLayers);
}

void Inference::InferLayers() {
  using namespace std::placeholders;

  // Convert the input from float to communicable
  ComVec input_com(input_.size());
  std::transform(std::par_unseq, input_.begin(), input_.end(),
      input_com.begin(), FloatToCom);

  // Iterate over all the layers and execute each one
  try {
    ComVec output_com = std::accumulate(layers_.begin(), layers_.end(),
        input_com, std::bind(&Layer::Infer, _2, _1));
  } catch (const Exception& e) {
    // WARNING I'm not sure std::accumulate propogates exceptions...
    result_.set_exception(e);
    return;
  }

  // Convert the output back to float
  std::vector<float> output(output_com.size());
  std::transform(output_com.begin(), output_com.end(), output.begin(),
      ComToFloat);
  result_.set_value(output);
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
          UnexpectedMessageError("Received another model share"));
      }
      break;
    case ServerMessage::VECTOR_SHARE: // WARNING This will be changed
      {
        std::scoped_lock<std::shared_mutex> layer_messages_lock
          (layer_messages_mutex_);
        layer_messages_.push(server_message);
      }
      layer_messages_condition_.notify_one();
      break;
  }
}

}  // namespace amrc
