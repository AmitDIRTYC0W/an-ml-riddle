// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_LAYER_H_
#define ANMLRIDDLE_CLIENT_LAYER_H_

#include <future>
#include <vector>

#include <capnp/message.h>

#include <anmlriddle/client/inference.h>
#include <anmlriddle/com.h>
#include "Model.capnp.h"

namespace amrc {

class Inference;

class Layer {
 protected:
      // TODO maybe it should be a weak_ptr? idk
    Inference& inference_;

    ComVec& get_result_share_();
    void set_result_share_(std::shared_ptr<ComVec> result_share);
    void set_model_share_(const ModelShare::Reader model_share);
    void Next();
    std::promise<std::vector<float>>& get_result_();
    void Send(const ::capnp::MessageBuilder& message);
 public:
    explicit Layer(Inference& inference) : inference_(inference) {}
    virtual ~Layer() = default;
    virtual void Begin() {}
    virtual void Receive(::capnp::MessageReader& message) {}
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_LAYER_H_
