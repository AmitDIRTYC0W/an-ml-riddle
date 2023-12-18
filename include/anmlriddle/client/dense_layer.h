// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_DENSE_LAYER_H_
#define ANMLRIDDLE_CLIENT_DENSE_LAYER_H_

#include <anmlriddle/client/layer.h>

#include "Model.capnp.h"

namespace amrc {

class DenseLayer : public Layer {
 private:
    LayerShare::Dense::Reader layer_share_;
 public:
    DenseLayer(Inference& inference, LayerShare::Dense::Reader layer_share)
        : Layer(inference), layer_share_(layer_share) {}
    void Begin();
    void Receive(::capnp::MessageReader& message) {};
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_DENSE_LAYER_H_
