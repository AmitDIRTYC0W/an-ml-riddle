#!/bin/sh

docker build -t anmlriddle_mnist .

mkdir -p ./artifacts
docker run -v "$(pwd)"/artifacts:/artifacts anmlriddle_mnist

