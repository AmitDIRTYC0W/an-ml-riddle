// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_FINAL_LAYER_H_
#define ANMLRIDDLE_CLIENT_FINAL_LAYER_H_

#include <capnp/message.h>

#include <anmlriddle/client/layer.h>

namespace amrc {

class FinalLayer : public Layer {
 public:
    explicit FinalLayer(Inference& inference) : Layer(inference) {}
    ComVec Infer(ComVec last);
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_FINAL_LAYER_H_
