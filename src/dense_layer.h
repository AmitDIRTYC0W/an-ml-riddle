// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_DENSE_LAYER_H_
#define ANMLRIDDLE_DENSE_LAYER_H_

#include <anmlriddle/com.h>
#include <anmlriddle/multiplication_triplet.h>
#include <anmlriddle/synchronised_queue.h>
#include "server_message_generated.h"

namespace amr {
         
unsigned short DenseLayerOutputSize(const DenseLayerShareT& share);

void InferDenseLayer(const DenseLayerShareT& share,
                     const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                     Eigen::Ref<Eigen::MatrixX<Com>> output, MTProvider get_mt,
                     IO<> io);

}  // namespace amr

#endif  // ANMLRIDDLE_DENSE_LAYER_H_
