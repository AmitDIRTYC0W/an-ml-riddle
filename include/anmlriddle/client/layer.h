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

    void Send(const ::capnp::MessageBuilder& message);
    ServerMessage::Reader FetchMessage();
 public:
    explicit Layer(Inference& inference) : inference_(inference) {}
    virtual ~Layer() = default;
    virtual ComVec Infer(ComVec last) = 0;
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_LAYER_H_
