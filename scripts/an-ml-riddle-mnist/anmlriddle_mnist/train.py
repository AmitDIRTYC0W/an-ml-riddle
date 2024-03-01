import random

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import pyswarms as ps
from sklearn.datasets import load_digits
import numpy as np

from . import com
from .model import Model
from .dense import Dense
from .relu import ReLU
from .softmax import Softmax

dataset = load_digits()

BATCH_SIZE = 16

def obtain_batch(size=BATCH_SIZE) -> tuple[ArrayLike, ArrayLike]:
    index = random.randint(0, len(dataset.data) - size)
    return dataset.data[index:index + size], dataset.target[index:index + size]

def define_model(parameters: ArrayLike, log: bool = False) -> Model:
    no_inputs = 8 * 8
    no_hidden = 16
    no_classes = 10
    
    W1 = parameters[0:1024].reshape((no_hidden, no_inputs))
    b1 = parameters[1024:1040].reshape((no_hidden,))
    W2 = parameters[1040:1200].reshape((no_classes, no_hidden))
    b2 = parameters[1200:1210].reshape((no_classes,))
    
    return Model([Dense(W1, b1),
                  ReLU(),
                  Dense(W2, b2),
                  Softmax(log)])

def evaluate(parameters: ArrayLike, input_: ArrayLike, class_: int) -> float:
    # Perform one-hot encoding. For example, 3 shall become [0, 0, MAX, 0, 0, ...]
    expected_output = jax.nn.one_hot(class_, 10) * com.MAXIMUM_VALUE

    # Evaluate the neural network
    model = define_model(parameters, log=True)
    actual_output = model.infer(input_)

    # Compute the cross entropy
    cost = jnp.dot(expected_output, actual_output)

    return cost

def objective(parameters: ArrayLike) -> float:
    batch_evaluate = jax.vmap(evaluate, in_axes=(None, 0, 0))
    images, image_classes = obtain_batch()
    return jnp.mean(batch_evaluate(parameters, images, image_classes))

batch_objective = jax.jit(jax.vmap(objective, in_axes=(0)))

def train_mnist_model() -> Model:
    dimensions = 1210
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds = (np.full(dimensions, com.MINIMUM_VALUE_COM), np.full(dimensions, com.MAXIMUM_VALUE_COM))
    optimiser = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions,
                                        options=options, bounds=bounds)

    cost, parameters = optimiser.optimize(batch_objective, iters=5000, n_processes=None)

    return define_model(parameters)

    # example_x, example_y = obtain_batch(1)
    # example_x = example_x[0]
    # example_y = example_y[0]

    # y = infer(parameters, example_x)
    # y_i = np.argmax(y)

    # print(y[y_i], y[example_y])

    # print(y_i, example_y)

    # print(y)
    
if __name__ == '__main__':
    train()
