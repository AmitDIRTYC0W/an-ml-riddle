// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_INFERENCE_H_
#define ANMLRIDDLE_CLIENT_INFERENCE_H_

#include <vector>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <future>

#include <capnp/message.h>

#include <anmlriddle/client/layer.h>
#include <anmlriddle/com.h>
#include "Model.capnp.h"
// #include <amrc/Model.capnp.h>
// #include "model_share_generated.h"

class Layer;

namespace amrc {

class Inference {
  friend class Layer;

 private:
    std::function<void(const ::capnp::MessageBuilder&)> send_;
    std::unique_ptr<Layer> current_layer_;
    capnp::List<LayerShare>::Reader::Iterator current_layer_it_;
    std::optional<ModelShare::Reader> model_share_;
    std::promise<std::vector<float>> result_;
    std::shared_ptr<ComVec> result_share_;

    void Next();

    inline void set_model_share_(ModelShare::Reader model_share) {
      model_share_ = model_share;
      current_layer_it_ = model_share.getLayers().begin();
    }

    // TODO Add optional parameter to Next to choose the next layer
    // that is not in the model

 public:
    Inference(std::function<void(const ::capnp::MessageBuilder&)> send,
              std::vector<float> input);

    void Receive(::capnp::MessageReader& message);

    inline std::promise<std::vector<float>>& get_result() {
      return result_;
    }
};

}  // namespace amrc

#endif
