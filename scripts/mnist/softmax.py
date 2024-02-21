import jax
from jax.typing import ArrayLike

from layer import Layer

class Softmax(Layer):
	def __init__(self, log: bool = False):
		self.log = log
		
	def infer(self, input_: ArrayLike) -> ArrayLike:
		if self.log:
			return jax.nn.log_softmax(input_)
		else:
			return jax.nn.softmax(input_)
