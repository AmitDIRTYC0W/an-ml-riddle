// Copyright 2023 Amit Goren

#include <anmlriddle/client/inference.h>

#include "../generate_shares.h"
#include <anmlriddle/client/initial_layer.h>
#include <anmlriddle/client/final_layer.h>
#include <anmlriddle/com.h>
#include <anmlriddle/client/dense_layer.h>

namespace amrc {

Inference::Inference(std::function<void(const ::capnp::MessageBuilder&)> send,
                     std::vector<float> input) : send_(send) {
  current_layer_ = std::make_unique<InitialLayer>(*this, input);
}

void Inference::Next() {
  if (++current_layer_it_ != model_share_.value().getLayers().end()) {
    switch (current_layer_it_->which()) {
      case LayerShare::DENSE:
        auto dense_layer_share = current_layer_it_->getDense();
        current_layer_ = std::make_unique<DenseLayer>(*this, dense_layer_share);
        break;
    }
  } else {
    current_layer_ = std::make_unique<FinalLayer>(*this);
  }
}

void Inference::Receive(::capnp::MessageReader& message) {
  current_layer_->Receive(message);
}

// void Inference::SetCurrentLayer(size_t index) {
//   current_layer_index_ = index;
//   switch (model_share_.getLayers()[index]) {
//     case ,,,
//   }
// }

// inline void Receive(::capnp::MessageReader& message) {
//   if (!model_share_) {
//     model_share_ = message.getRoot<ModelShare>();
//     current_layer_ = 
//   } else {
//     state_->Receive(*this, message);
//   }
// }

// void Inference::Begin(std::vector<float> input) {
//   state_ = std::make_unique<NoModelInferenceState>();
// }

}  // namespace amrc
