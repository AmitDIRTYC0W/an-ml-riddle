import functools

from jax.typing import ArrayLike
import jax.numpy as jnp
import flatbuffers

from .layer import Layer, SerializableLayer
from .anmlriddle import Model as FlatBuffersModel

class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def infer(self, input_: ArrayLike) -> ArrayLike:
        activations_com = jnp.int16(input_ * 16)

        for layer in self.layers:
            activations_com = layer.infer(activations_com)

        return activations_com

    def serialize(self, builder: flatbuffers.Builder) -> FlatBuffersModel:
        is_serializable = lambda layer: issubclass(type(layer), SerializableLayer)
        serializable_layers = filter(is_serializable, self.layers)

        serialized_layers = list(map(lambda layer: layer.serialize(builder), serializable_layers))

        FlatBuffersModel.ModelStartLayersVector(builder, len(serialized_layers))
        for layer in reversed(serialized_layers):
            builder.PrependUOffsetTRelative(layer)
        serialized_layers_vector = builder.EndVector()

        FlatBuffersModel.ModelStart(builder)
        FlatBuffersModel.ModelAddLayers(builder, serialized_layers_vector)
        return FlatBuffersModel.ModelEnd(builder)

