// Copyright 2024 Amit Goren

#include "split_model.h"

#include <anmlriddle/com.h>
#include "../dense_layer.h"

namespace anmlriddle {

void SplitModel(const Model* model, std::pair<ModelShareT*, flatbuffers::DetachedBuffer*> shares) {
  flatbuffers::FlatBufferBuilder builder;
  
  auto no_layers = model->layers()->size();
  
  // Prepare vectors of layer shares
  shares.first->layerShares = std::vector<LayerShareUnion>(no_layers);
  std::vector<flatbuffers::Offset<void>> second_layer_shares(no_layers);
  std::vector<uint8_t> layer_share_types(no_layers); 
  
  // Split each layer
  auto layer = model->layers()->cbegin();
  auto layer_type = model->layers_type()->cbegin();
  
  auto first_layer_share = shares.first->layerShares.begin();
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

}  // namespace anmlriddle
