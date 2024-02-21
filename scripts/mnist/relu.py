import jax.numpy as jnp
from jax.typing import ArrayLike

from layer import Layer

class ReLU(Layer):
	def infer(self, input_: ArrayLike) -> ArrayLike:
		return jnp.maximum(0, input_)
