#include "infer_layer.h"

#include "dense_layer.h"

namespace amr {

/*template <typename Derived>
Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::MatrixBase<Derived>& input,
                               MTProvider get_mt, IO<> io) {*/
Eigen::VectorX<Com> InferLayer(const LayerShareUnion layer,
                               const Eigen::Ref<const Eigen::MatrixX<Com>>& input,
                               MTProvider get_mt, IO<> io) {
  switch (layer.type) {
   case LayerShare_DenseLayerShare: {
    auto dense_layer = layer.AsDenseLayerShare();
    Eigen::VectorX<Com> output(DenseLayerOutputSize(*dense_layer));
    InferDenseLayer(*dense_layer, input, output, get_mt, io);
    return output;
   }
   default:
    throw std::runtime_error("InferLayer: unknown layer type");
  }
}

}  // namespace amr