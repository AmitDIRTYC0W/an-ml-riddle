#!/bin/sh

docker build -t jax_mnist .

# artifacts?
mkdir -p ./output
docker run -v "$(pwd)"/output:/output jax_mnist

