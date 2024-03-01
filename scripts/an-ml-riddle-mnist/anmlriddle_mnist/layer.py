from abc import ABC, abstractmethod

from jax.typing import ArrayLike
import flatbuffers

class Layer(ABC):
	@abstractmethod
	def infer(self, input_: ArrayLike) -> ArrayLike:
		pass
	
class SerializableLayer(Layer):
	@abstractmethod
	def serialize(self, builder: flatbuffers.Builder) -> Layer:
		pass
