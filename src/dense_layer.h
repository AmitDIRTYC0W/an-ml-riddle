// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_DENSE_LAYER_H_
#define ANMLRIDDLE_DENSE_LAYER_H_

#include <anmlriddle/com.h>
#include <anmlriddle/multiplication_triplet.h>
#include "model_generated.h"
#include "server_message_generated.h"

namespace anmlriddle {
         
unsigned short DenseLayerOutputSize(const DenseLayerShareT& share);
unsigned short DenseLayerOutputSize(const DenseLayerShare* share);

void InferDenseLayer(const DenseLayerShareT& share,
                     const Eigen::Ref<const Eigen::MatrixX<Com>> input,
                     Eigen::Ref<Eigen::MatrixX<Com>> output, MTProvider get_mt,
                     SingularSink<MTInferenceShare>& messagesSink, SendFunction send,
                     std::stop_token stop_token);

void InferDenseLayer(const DenseLayerShare* share,
                     const Eigen::Ref<const Eigen::MatrixX<Com>> input,
                     Eigen::Ref<Eigen::MatrixX<Com>> output, MTProvider get_mt,
                     SingularSink<MTInferenceShare>& messagesSink, SendFunction send,
                     std::stop_token stop_token);


using LayerSharePair = std::pair<LayerShareUnion, flatbuffers::Offset<void>>; // TODO move to another file

LayerSharePair SplitDenseLayer(const DenseLayer* layer, flatbuffers::FlatBufferBuilder& builder);

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_DENSE_LAYER_H_
