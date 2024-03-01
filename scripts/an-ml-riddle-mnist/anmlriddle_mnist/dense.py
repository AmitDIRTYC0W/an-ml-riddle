from jax.typing import ArrayLike
import jax.numpy as jnp
import flatbuffers

from . import com
from .layer import SerializableLayer
from .utils import serialize_matrix
from .anmlriddle import DenseLayer

class Dense(SerializableLayer):
	def __init__(self, weights: ArrayLike, biases: ArrayLike):
		self.weights = weights
		self.biases = biases

	def infer(self, input_: ArrayLike) -> ArrayLike:
		res = jnp.matmul(self.weights, input_) // (1 << com.FRACTION_BITS) + self.biases
		return res

	def serialize(self, builder: flatbuffers.Builder) -> DenseLayer:
		serialized_biases = builder.CreateNumpyVector(self.biases)
		serialized_weights = serialize_matrix(builder, self.weights)

		DenseLayer.DenseLayerStart(builder)
		DenseLayer.DenseLayerAddWeights(builder, serialized_weights)
		DenseLayer.DenseLayerAddBiases(builder, serialized_biases)
		return DenseLayer.DenseLayerEnd(builder)
