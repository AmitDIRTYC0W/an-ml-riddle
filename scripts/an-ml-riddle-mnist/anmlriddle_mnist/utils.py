from jax.typing import ArrayLike
import flatbuffers

from .anmlriddle import Matrix

def serialize_matrix(builder: flatbuffers.Builder, matrix: ArrayLike) -> Matrix:
    values = matrix.flatten(order='C')
    serialized_values = builder.CreateNumpyVector(values)

    Matrix.MatrixStart(builder)
    Matrix.MatrixAddValues(builder, serialized_values)
    Matrix.MatrixAddColumns(builder, matrix.shape[1])
    return Matrix.MatrixEnd(builder)
