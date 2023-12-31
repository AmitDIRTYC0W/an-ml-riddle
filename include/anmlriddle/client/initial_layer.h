// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_INITIAL_LAYER_H_
#define ANMLRIDDLE_CLIENT_INITIAL_LAYER_H_

#include <vector>

#include <anmlriddle/client/layer.h>
#include <anmlriddle/client/unexpected_message_error.h>

namespace amrc {

class InitialLayer : public Layer {
 public:
    explicit InitialLayer(Inference& inference) : Layer(inference) {}
    virtual ~InitialLayer() = default;  // TODO remove 'virtual'
    ComVec Infer(ComVec last);
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_INITIAL_LAYER_H_
