#ifndef ANMLRIDDLE_INFER_LAYER_H_
#define ANMLRIDDLE_INFER_LAYER_H_

#include <Eigen/Dense>

#include "server_message_generated.h"
#include <anmlriddle/multiplication_triplet.h>
#include <anmlriddle/synchronised_queue.h>

namespace amr {

/*Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               Eigen::Ref<const Eigen::MatrixX<Com>> input,
                               MTProvider get_mt, IO<> io);*/

/*
template <typename Derived>
Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::MatrixBase<Derived>& input,
                               MTProvider get_mt, IO<> io);*/

Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                               MTProvider get_mt, IO<> io);

}  // namespace amr

#endif  // ANMLRIDDLE_INFER_LAYER_H_