import functools

from jax.typing import ArrayLike
import jax.numpy as jnp

from layer import Layer

class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def infer(self, input_: ArrayLike) -> ArrayLike:
        activations_com = jnp.int16(input_ * 16)

        for layer in self.layers:
            activations_com = layer.infer(activations_com)

        return activations_com
