// Copyright 2024 Amit Goren

#include "split_model.h"

#include <anmlriddle/com.h>
#include "../split_vector.h"
#include <ranges>

/*
inline void SplitDenseLayer(LayerDescription::Dense::Reader& description,
                            std::pair<LayerShare::Dense::Builder,
                               LayerShare::Dense::Builder> shares) {
  auto biases = description.getBiases();

  auto biases_shares = std::pair(AsSpan(shares.first.getBiasesShare()),
      AsSpan(shares.seco

  SplitSpan(biases, biases_shares);
}
*/

namespace amr {

using LayerSecret = std::pair<LayerShareUnion, flatbuffers::Offset<void>>;

inline LayerSecret SplitDenseLayer(
    const DenseLayer* layer,
    flatbuffers::FlatBufferBuilder& builder) {
  std::size_t output_size = layer->biases()->size();
  std::size_t weights_size = layer->weights()->values()->size();
  #warning add assertions wherever output_size is used
 
  // WARNING This must be deleted manually
  DenseLayerShareT* first_share = new DenseLayerShareT();
  
  // Split the biases into shares
  first_share->biasesShare = std::vector<Com>(output_size);
  auto [second_biases_share_flatbuffers, second_biases_share_eigen] = FlatbuffersComVector(output_size, builder);
  Eigen::Map<const Eigen::VectorX<Com>> secret_biases_eigen(layer->biases()->data(), output_size);
  Eigen::Map<Eigen::VectorX<Com>> first_biases_share_eigen(first_share->biasesShare.data(), output_size);
  SplitDense<Eigen::VectorX<Com>>(secret_biases_eigen, std::make_pair(first_biases_share_eigen, second_biases_share_eigen));

  // Split the weights into share
  first_share->weightsShare = std::make_unique<MatrixT>();
  first_share->weightsShare->values = std::vector<Com>(weights_size);
  unsigned short weights_rows = RowsInMatrix(weights_size, layer->weights()->columns());
  auto [second_weights_share_flatbuffers, second_weights_share_eigen] = FlatbuffersComMatrix(
      weights_rows, layer->weights()->columns(), builder);
  auto secret_weights_eigen = AsComMatrix(layer->weights());
  auto first_weights_share_eigen = AsComMatrix(*(first_share->weightsShare));
  SplitDense<Eigen::MatrixX<Com>>(secret_weights_eigen, std::make_pair(first_weights_share_eigen, second_weights_share_eigen));
  
  auto second_share = CreateDenseLayerShare(builder, second_weights_share_flatbuffers,
                                            second_biases_share_flatbuffers);
  
  LayerShareUnion first_share_union;
  first_share_union.type = LayerShare_DenseLayerShare;
  first_share_union.value = first_share;
  
  return std::make_pair(first_share_union, second_share.Union());
}

void SplitModel(const Model* model, std::pair<ModelShareT*, flatbuffers::DetachedBuffer*> shares) {
  flatbuffers::FlatBufferBuilder builder;
  
  auto no_layers = model->layers()->size();
  
  // Prepare vectors of layer shares
  shares.first->layers = std::vector<LayerShareUnion>(no_layers);
  std::vector<flatbuffers::Offset<void>> second_layer_shares(no_layers);
  std::vector<uint8_t> layer_share_types(no_layers); 
  
  // Split each layer
  auto layer = model->layers()->cbegin();
  auto layer_type = model->layers_type()->cbegin();
  
  auto first_layer_share = shares.first->layers.begin();
  auto second_layer_share = second_layer_shares.begin();
  auto layer_share_type = layer_share_types.begin();
  
  while (layer != model->layers()->cend()) {
    switch (*layer_type) {
     case Layer_DenseLayer: {
      std::tie(*first_layer_share, *second_layer_share) = SplitDenseLayer(
          static_cast<const DenseLayer*>(*layer), builder);
      break;
     }
     default: {
      throw std::runtime_error("SplitModel: unknown layer type");
     }
    }
    
    // If you are reading this, just know there are more iterators here than bitches in your life
    ++layer;
    ++layer_type;
    ++first_layer_share;
    ++second_layer_share;
    ++layer_share_type;
  }
  
  // Pack the second share into a ready-to-send buffer
  auto second_model_share = CreateModelShareDirect(builder, &layer_share_types, &second_layer_shares);
  auto message = CreateServerMessage(builder, ServerMessageUnion_ModelShare, second_model_share.Union());
  builder.Finish(message);
  *shares.second = builder.Release();
}

}  // namespace amr
