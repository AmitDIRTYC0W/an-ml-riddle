#ifndef ANMLRIDDLE_INFER_LAYER_H_
#define ANMLRIDDLE_INFER_LAYER_H_

#include <Eigen/Dense>

#include "server_message_generated.h"
#include <anmlriddle/multiplication_triplet.h>

namespace anmlriddle {

/*
template <typename Derived>
Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::MatrixBase<Derived>& input,
                               MTProvider get_mt, IO<> io);*/

Eigen::VectorX<Com> InferLayer(const void* layer,
                               const LayerShare layer_type,
                               const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                               MTProvider get_mt, std::stop_token stop_token,
                               SingularSink<MTInferenceShare>& messaegesSink,
                               SendFunction send);

Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                               MTProvider get_mt, std::stop_token stop_token,
                               SingularSink<MTInferenceShare>& messaegesSink,
                               SendFunction send);


}  // namespace anmlriddle

#endif  // ANMLRIDDLE_INFER_LAYER_H_