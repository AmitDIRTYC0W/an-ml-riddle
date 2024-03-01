import os

import flatbuffers
import numpy as np

from .dense import Dense
from .relu import ReLU
from .softmax import Softmax
from .model import Model
from .train import train_mnist_model

model_file_identifier = b'AMRM'

def _save_model_to_file(model: Model, filepath: str) -> None:
    builder = flatbuffers.Builder(1024)
    serialized_model = model.serialize(builder)
    builder.Finish(serialized_model, model_file_identifier)

    with open(filepath, 'wb') as f:
        f.write(builder.Output())

def main():
    # os.system('ls /usr/local/lib/python3.11/site-packages/anmlriddle_mnist/anmlriddle')
    # Matrix.MatrixStartValuesVector(builder, 9)
    # for i in reversed(range(10)):
    #     builder.PrependInt16(i)
    # values = builder.EndVector()

    # Matrix.MatrixStart(builder)
    # Matrix.MatrixAddColumns(builder, 3)
    # matrix = Matrix.MatrixEnd(builder)

    print('Hello, world!')
    model = train_mnist_model()
    _save_model_to_file(model, '/artifacts/model.amrm')

