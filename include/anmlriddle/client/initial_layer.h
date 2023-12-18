// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_INITIAL_LAYER_H_
#define ANMLRIDDLE_CLIENT_INITIAL_LAYER_H_

#include <vector>

#include <anmlriddle/client/layer.h>
#include <anmlriddle/client/unexpected_message_error.h>

namespace amrc {

class InitialLayer : public Layer {
 private:
    std::vector<float> input_;
 public:
    InitialLayer(Inference& inference, std::vector<float> input)
        : Layer(inference), input_(input) {}
    virtual ~InitialLayer() = default;  // TODO remove 'virtual'
    void Receive(::capnp::MessageReader& message);
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_INITIAL_LAYER_H_
