from __future__ import annotations

from typing import Union

import tensorflow as tf
from tensorflow import Tensor, RaggedTensor, SparseTensor


def flatten_at_dim(tensor: Tensor | RaggedTensor, axis=0) -> Tensor | RaggedTensor:
    with tf.name_scope('flatten_at_dim'):
        rank = tf.rank(tensor)
        axis = rank + axis if tf.less(axis, 0) else axis
        print(rank)
        print(axis)

        old_shape = tf.shape(tensor)
        try:
            new_dimension = tf.gather(old_shape, axis) * tf.gather(old_shape, axis + 1)
        except TypeError:
            new_dimension = -1
        pre_shape = tf.slice(old_shape, [0], [axis])
        post_shape = tf.slice(old_shape, [axis + 2], [-1])
        new_shape = tf.concat([pre_shape, [new_dimension], post_shape], axis=0)
        return tf.reshape(tensor, new_shape)


def expand_tensor_leftwards_to_rank_of_tensor(
        tensor: Union[Tensor, RaggedTensor],
        to_tensor: Union[Tensor, SparseTensor, RaggedTensor]) -> Tensor:
    '''
    Expands the rank of `tensor` to the rank of `to_tensor`.
    :param tensor: The tensor whose rank the function is to expand.
    :param to_tensor: The tensor whose rank the first given tensor should be expanded to.
    :return:
    '''
    try:
        tf.assert_less(tf.rank(tensor), tf.rank(to_tensor))
        rank_difference = tf.rank(to_tensor) - tf.rank(tensor)
        expanded_dims = tf.ones_like(tf.slice(tf.shape(to_tensor), [0], [rank_difference]))
        new_tensor_shape = tf.concat((expanded_dims, tf.shape(tensor)), axis=0)
        try:
            return tf.reshape(tensor, new_tensor_shape)
        except TypeError:
            return tf.sparse.reshape(tensor, new_tensor_shape)
    except tf.errors.InvalidArgumentError:
        return tensor


def expand_batched_tensor_leftwards_to_rank_of_tensor(
        batched_tensor: Union[Tensor, RaggedTensor],
        to_tensor: Union[Tensor, SparseTensor, RaggedTensor]) -> Tensor:
    '''
    Expands the rank of `batched_tensor` to the rank of `to_tensor`. The first dimension of `batched_tensor` is
    assumed to be the batch dimension.
    :param batched_tensor: The tensor whose rank the function is to expand.
    :param to_tensor: The tensor whose rank the first given tensor should be expanded to.
    :return:
    '''
    try:
        tf.assert_less(len(batched_tensor.shape), len(to_tensor.shape))
        rank_difference = len(to_tensor.shape) - len(batched_tensor.shape)
        expanded_dims = (1,) * rank_difference
        new_tensor_shape = batched_tensor.shape[0:1] + expanded_dims + batched_tensor.shape[1:]
        try:
            return tf.reshape(batched_tensor, new_tensor_shape)
        except TypeError:
            return tf.sparse.reshape(batched_tensor, new_tensor_shape)
    except tf.errors.InvalidArgumentError:
        return batched_tensor
