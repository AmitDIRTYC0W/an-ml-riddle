#include "infer_layer.h"

#include "anmlriddle/channel.h"
#include "common_generated.h"
#include "dense_layer.h"
#include "server_message_generated.h"
#include <stop_token>

namespace anmlriddle {

/*template <typename Derived>
Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::MatrixBase<Derived>& input,
                               MTProvider get_mt, IO<> io) {*/
Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                               MTProvider get_mt, std::stop_token stop_token, SingularSink<MTInferenceShare>& messagesSink, SendFunction send) {
  switch (layer.type) {
   case LayerShare_DenseLayerShare: {
    auto dense_layer = layer.AsDenseLayerShare();
    Eigen::VectorX<Com> output(DenseLayerOutputSize(*dense_layer));
    InferDenseLayer(*dense_layer, input, output, get_mt, messagesSink, send, stop_token);
    return output;
   }
   default:
    throw std::runtime_error("InferLayer: unknown layer type");
  }
}

Eigen::VectorX<Com> InferLayer(const void* layer, const LayerShare layer_type,
                               const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                               MTProvider get_mt, std::stop_token stop_token, SingularSink<MTInferenceShare>& messagesSink, SendFunction send) {
  switch (layer_type) {
   case LayerShare_DenseLayerShare: {
    const DenseLayerShare* dense_layer = static_cast<const DenseLayerShare*>(layer);
    Eigen::VectorX<Com> output(DenseLayerOutputSize(dense_layer));
    InferDenseLayer(dense_layer, input, output, get_mt, messagesSink, send, stop_token);
    return output;
   }
   default:
    throw std::runtime_error("InferLayer: unknown layer type");
  }
}

}  // namespace anmlriddle