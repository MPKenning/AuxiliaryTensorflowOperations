from typing import Union

import tensorflow as tf
from numpy import ndarray

from tensorflow import Tensor, SparseTensor
from tensorflow._api.v2 import sparse as tf_sparse
from tensorflow.python.ops.check_ops import assert_equal, assert_rank_at_least
from tensorflow.python.ops.sparse_ops import from_dense


def sparse_square_diag(tensor: SparseTensor) -> SparseTensor:
    """
    For a SparseTensor of any rank, this function returns a diagonal matrix based on the innermost dimension.
    I.e., for a SpareTensor with shape [..., a], it will return [..., a, a].
    :param tensor:
    :return:
    """
    # Compute the number of elements in the diagonal.
    n_diag_elems = tf.gather(tf.shape(tensor), tf.rank(tensor) - 1)

    # Stack coords and transpose.
    coords = tensor.indices[:, -1:]
    other_coords = tensor.indices[:, :-1]
    indices = tf.concat([other_coords, coords, coords], axis=-1)

    # Get the values.
    values = tensor.values

    # Make the shape.
    shape = tf.cast(tf.concat([tf.shape(tensor), [n_diag_elems]], axis=0), dtype=tf.int64)

    # Make the diagonal tensor.
    diag_tensor = SparseTensor(indices, values, shape)

    return diag_tensor


def sparse_flatten_at_dim(tensor: SparseTensor, axis=0) -> SparseTensor:
    axis = tf.rank(tensor) + axis if axis < 0 else axis
    if axis == tf.rank(tensor) - 1:
        return tensor
    elif axis >= tf.rank(tensor):
        raise IndexError('An axis was given that does not exist.')

    original_shape = tf.shape(tensor)
    shape_left_of_flatten = original_shape[0:axis]
    to_flatten = original_shape[axis:axis + 2]
    shape_right_of_flatten = original_shape[axis + 2:]
    flattened_dims = tf.reduce_prod(to_flatten)
    try:
        new_shape = tf.concat((shape_left_of_flatten, [flattened_dims], shape_right_of_flatten), axis=0)
    except TypeError:
        new_shape = tf.concat((shape_left_of_flatten, [-1], shape_right_of_flatten), axis=0)
    return tf.sparse.reshape(tensor, new_shape)


def sparse_squeeze(tensor: SparseTensor, axis: int = 0):
    presqueeze_shape = tensor.shape
    assert_equal(presqueeze_shape[axis], 1)
    squeeze_to = presqueeze_shape.as_list()
    del squeeze_to[axis]
    squeezed_tensor = tf_sparse.reshape(tensor, squeeze_to)
    return squeezed_tensor


def sparse_dense_batched_matmul(tensor_a: Union[Tensor, SparseTensor], tensor_b: Union[Tensor, SparseTensor],
                                transpose_sparse=False, transpose_dense=False) -> Union[Tensor, SparseTensor]:
    """
    Computes the matrix multiplication of a sparse tensor and a dense tensor. There is no order to the first two
    tensor arguments to this function. It is assumed that the dense tensor has a lower rank than the sparse tensor
    and that its dimensions match the sparse tensor's inner dimensions. For example:
        dense: [b, c], sparse: [d, e, a, b].

    The dense tensor should be able to be braodcast onto the sparse tensor's shape. Its shape should be consistent with
    the inner dimensions of the sparse tensor. For example:
        dense: [a, b, c], sparse: [..., a, b, c].

    The following is not currently allowed:
        dense: [c, 1, a, b], sparse: [c, d, a, b].

    The dense tensor will be expanded and tiled to match the sparse tensor's dimensions.

    :param tensor_a: The first tensor, either SparseTensor or Tensor.
    :param tensor_b: The second tensor, either SparseTensor or Tensor.
    :param transpose_sparse: (Default: True) Whether the sparse tensor needs to be transposed first. If both tensors
    are dense, the first tensor will be transposed or not according to this parameter, behaving like transpose_a.
    :param transpose_dense: (Default: True) Whether the dense tensor needs to be transposed first. If both tensors
    are dense, the first tensor will be transposed or not according to this parameter, behaving like transpose_b.
    :return:
    """

    sparse_is_on_left = False
    # If both tensors are dense, pass them to the library function.
    if _both_tensors_are_dense(tensor_a, tensor_b):
        return tf.matmul(tensor_a, tensor_b, transpose_a=transpose_sparse, transpose_b=transpose_dense)

    # If both tensors are sparse, pass them to the batch matmul.
    elif _both_tensors_are_sparse(tensor_a, tensor_b) and _tensors_share_the_same_rank(tensor_a, tensor_b):
        return sparse_batched_matmul(tensor_a, tensor_b, transpose_a=transpose_sparse, transpose_b=transpose_dense)

    # Otherwise find out which one is dense.
    elif isinstance(tensor_b, SparseTensor) and not isinstance(tensor_a, SparseTensor):
        dense, sparse = tensor_a, tensor_b

    # Right one must be dense, etc.
    else:
        sparse_is_on_left = True
        sparse, dense = tensor_a, tensor_b

    # Ensure that the dense tensor has at least two dimensions.
    assert_rank_at_least(dense, 2)

    with tf.name_scope('sparse_dense_batched_matmul'):
        # Expand the dims of the dense matrix starting at the third from last.
        while tf.less(tf.rank(dense), tf.rank(sparse)):
            # Expand the dimension.
            dense = tf.expand_dims(dense, axis=0)

            # Tile the dimension to the size of the corresponding dimension in the sparse tensor.
            dim_size_in_sparse = tf.gather(tf.shape(sparse), tf.rank(sparse) - tf.rank(dense))
            multiples = _make_multiples_tensor(tf.shape(dense), dim_size_in_sparse, 0)

            dense = tf.tile(dense, multiples)

        sparse.shape[:-2].assert_is_compatible_with(dense.shape[:-2])

        dense = from_dense(dense)

        if sparse_is_on_left:
            return sparse_batched_matmul(sparse, dense, transpose_a=transpose_sparse, transpose_b=transpose_dense)
        else:
            return sparse_batched_matmul(dense, sparse, transpose_a=transpose_dense, transpose_b=transpose_sparse)


def sparse_batched_matmul(tensor_a: SparseTensor, tensor_b: SparseTensor,
                          transpose_a=False, transpose_b=False) -> SparseTensor:
    """
    Performs a matrix multiplication on two given `SparseTensors`. They should have a rank ≥ 2 and their rank should
    be equal. Their innermost dimensions must be identical. Their outermost two dimensions must also have a compatible
    intermediate dimension. For example,
        compatible shapes: a.shape: [..., d, e] and b.shape: [..., e, f] before b is transposed.
    By default, `tensor_b` is assumed not to be transposed; i.e., it will need to be transposed within the function
    before being multiplied with `tensor_a`.
    :param tensor_a: A SparseTensor of rank ≥ 2.
    :param tensor_b: A SparseTensor fo rank ≥ 2.
    :param transpose_a: (Bool, default: True) Whether `tensor_a`'s two outermost dimensions need to be transposed
    before multiplying the two tensors.
    :param transpose_b: (Bool, default: True) Whether `tensor_b`'s two outermost dimensions need to be transposed
    before multiplying the two tensors.
    :return:
    """
    with tf.name_scope('sparse_batched_matmul'):
        # Ensure that the two tensors are of identical rank.
        tensor_a.shape.assert_same_rank(tensor_b.shape)

        # Ensure that they are compatible.
        tensor_a.shape[:-2].assert_is_compatible_with(tensor_b.shape[:-2])

        # Ensure that their rank is at least two.
        tensor_a.shape.with_rank_at_least(2)

        # Transpose the dimensions as required.
        def transpose_outer_dimensions(tensor):
            rank = tf.rank(tensor)

            # Change the dimensions according to the rank.
            perm = tf.concat([tf.range(rank - 2), [rank - 1, rank - 2]], axis=0)

            # Reshape tensor_b to the new shape.
            return tf.sparse.transpose(tensor, perm)

        # Trying to mimic matmul here, so the operation makes no sense.
        tensor_a = transpose_outer_dimensions(tensor_a) if transpose_a else tensor_a
        tensor_b = transpose_outer_dimensions(tensor_b) if transpose_b else tensor_b

        # Determine the tiling factors. The factor of tensor a will be used to tile tensor b and vice versa.
        a_factor = tf.shape(tensor_a)[-2]
        b_factor = tf.shape(tensor_b)[-1]

        # Expand the dimensions.
        tensor_a = tf.sparse.expand_dims(tensor_a, axis=-1)
        tensor_b = tf.sparse.expand_dims(tensor_b, axis=-3)

        # Tile the respective dimensions.
        tensor_a = sparse_tile_on_axis(tensor_a, -1, b_factor)
        tensor_b = sparse_tile_on_axis(tensor_b, -3, a_factor)

        # Add the indices of each to the other to allow `map_values` to do its magic.
        tensor_a = tf.sparse.add(tensor_a, sparse_zeros_like(tensor_b))
        tensor_b = tf.sparse.add(tensor_b, sparse_zeros_like(tensor_a))

        # Multiply the two tensors.
        intermediate_result = tf.sparse.map_values(tf.multiply, tensor_a, tensor_b)

        # Then reduce the intermediate dimension.
        result = tf.sparse.reduce_sum(intermediate_result, axis=-2, output_is_sparse=True)

        return result


def sparse_zeros_like(sparse_tensor: SparseTensor) -> SparseTensor:
    """
    Returns a SparseTensor shaped like the given SparseTensor but consists of only zeros.
    :param sparse_tensor: The SparseTensor whose shape should be copied.
    :return:
    """
    indices = sparse_tensor.indices
    values = tf.zeros_like(sparse_tensor.values)
    shape = tf.cast(tf.shape(sparse_tensor), tf.int64)
    return tf.sparse.SparseTensor(indices, values, shape)


def sparse_ones_like(sparse_tensor: SparseTensor) -> SparseTensor:
    """
    Returns a SparseTensor shaped like the given SparseTensor but consists of only ones.
    :param sparse_tensor: The SparseTensor whose shape should be copied.
    :return:
    """
    indices = sparse_tensor.indices
    values = tf.ones_like(sparse_tensor.values)
    shape = tf.cast(tf.shape(sparse_tensor), tf.int64)
    return tf.sparse.SparseTensor(indices, values, shape)


def sparse_tile_on_axis(sparse_tensor: SparseTensor, axis, multiple) -> SparseTensor:
    """
    A tile operation that works on SparseTensors. The argument multiples is supposed to function like tf.tile's
    `multiples` argument. This is much faster than tf.sparse.concat if one is trying to tile a dimension.
    :param sparse_tensor: The SparseTensor to tile.
    :param axis: The axis to tile.
    :param multiple: The multiple by which to tile the given axis.
    :return:
    """
    rank = tf.rank(sparse_tensor)

    axis = rank + axis if axis < 0 else axis

    axis_to_original_pos_perm = tf.range(0)
    if axis > 0:
        # Make the first permutation of the data---bring selected axis to outermost dimension.
        original_perm = tf.range(rank)
        axis_to_outermost_perm = tf.concat(([axis], original_perm[:axis], original_perm[axis + 1:]), axis=0)

        # Make the last permutation of the data---bring selected axis back to its position from outermost dimension.
        axis_to_original_pos_perm = tf.concat((tf.range(1, axis + 1), [0], tf.range(axis + 1, rank)), 0)

        transposed_sparse_tensor = tf.sparse.transpose(sparse_tensor, axis_to_outermost_perm)
        transposed_sparse_tensor = tf.sparse.reorder(transposed_sparse_tensor)
    else:
        transposed_sparse_tensor = sparse_tensor

    # Compute the original shape.
    original_shape = tf.shape(transposed_sparse_tensor)

    # Multiply the original shape by the tiling multiple.
    multiples = _make_multiples_tensor(original_shape, multiple, 0)
    new_shape = tf.cast(original_shape, dtype=tf.int64) * tf.cast(multiples, dtype=tf.int64)

    # The sparse tensor's indices and values.
    indices = tf.tile(transposed_sparse_tensor.indices, [multiple, 1])
    values = tf.tile(transposed_sparse_tensor.values, [multiple])

    # Scale the values of the first column of indices.
    length_of_axis = tf.gather(original_shape, 0)
    n_entries = tf.gather(tf.shape(sparse_tensor.indices), 0)
    indices_scale = tf.cast(tf.repeat(tf.range(multiple) * length_of_axis, n_entries), tf.int64)
    indices_scale = tf.expand_dims(indices_scale, axis=0)
    indices = tf.transpose(tf.tensor_scatter_nd_add(tf.transpose(indices), [[0]], indices_scale))

    # Create the tiled sparse tensor.
    tiled_sparse_tensor = tf.sparse.SparseTensor(indices, values, new_shape)

    if axis > 0:
        sparse_tensor_transposed_back = tf.sparse.transpose(tiled_sparse_tensor, axis_to_original_pos_perm)
    else:
        sparse_tensor_transposed_back = tiled_sparse_tensor

    return sparse_tensor_transposed_back


def _both_tensors_are_sparse(tensor1: Union[ndarray, Tensor, SparseTensor],
                             tensor2: Union[ndarray, Tensor, SparseTensor]):
    return isinstance(tensor1, SparseTensor) and isinstance(tensor2, SparseTensor)


def _both_tensors_are_dense(tensor1: Union[ndarray, Tensor, SparseTensor],
                            tensor2: Union[ndarray, Tensor, SparseTensor]):
    return isinstance(tensor1, (ndarray, Tensor)) and isinstance(tensor2, (ndarray, Tensor))


def _tensors_share_the_same_rank(*tensors: Union[Tensor, SparseTensor]):
    for t in tensors[1:]:
        if tf.equal(tf.rank(t), tensors[0]):
            return False
    return True


def _make_multiples_tensor(tensor_like, multiple, axis):
    return tf.tensor_scatter_nd_update(tf.ones_like(tensor_like), [[axis]], [multiple])
