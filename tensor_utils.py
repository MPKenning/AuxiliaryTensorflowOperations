from __future__ import annotations

import tensorflow as tf
from tensorflow import Tensor, RaggedTensor


def flatten_at_dim(tensor: Tensor | RaggedTensor, axis=0) -> Tensor | RaggedTensor:
    axis = tf.rank(tensor) + axis if axis < 0 else axis
    if axis == tf.rank(tensor) - 1:
        return tensor
    elif axis >= tf.rank(tensor):
        raise IndexError('An axis was given that does not exist.')

    try:
        return tf.concat(tf.unstack(tensor, axis=axis), axis=axis)
    except ValueError:
        old_shape = tensor.shape
        try:
            new_dimension = old_shape[axis] * old_shape[axis+1]
        except TypeError:
            new_dimension = -1
        new_shape = tf.concat([old_shape[:axis], [new_dimension], old_shape[axis+2:]], axis=0)
        return tf.reshape(tensor, new_shape)