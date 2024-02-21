from abc import ABC, abstractmethod

from jax.typing import ArrayLike

class Layer(ABC):
	@abstractmethod
	def infer(self, input_: ArrayLike) -> ArrayLike:
		pass
