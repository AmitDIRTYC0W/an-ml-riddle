// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_SERVER_SPLIT_MODEL_H_
#define ANMLRIDDLE_SERVER_SPLIT_MODEL_H_

#include "model_generated.h"
#include "server_message_generated.h"

// NOTE Is it even needed?
/*
void SplitDenseLayer(LayerDescription::Dense::Reader description,
                     std::pair<LayerShare::Dense::Builder,
                               LayerShare::Dense::Builder> shares);
*/

namespace amr {

void SplitModel(const Model* model,
                std::pair<ModelShareT*, flatbuffers::DetachedBuffer*> shares);

}  // namespace amr

#endif  // ANMLRIDDLE_SERVER_SPLIT_MODEL_H_
