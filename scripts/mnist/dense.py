from jax.typing import ArrayLike
import jax.numpy as jnp

import com
from layer import Layer

class Dense(Layer):
	def __init__(self, weights: ArrayLike, biases: ArrayLike):
		self.weights = weights
		self.biases = biases

	def infer(self, input_: ArrayLike) -> ArrayLike:
		# return jnp.dot(self.weights, input_) // (1 << com.FRACTION_BITS) + self.biases
		res = jnp.matmul(self.weights, input_) // (1 << com.FRACTION_BITS) + self.biases
		return res
