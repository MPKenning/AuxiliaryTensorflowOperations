from typing import Union

import tensorflow as tf
from numpy import ndarray

from tensorflow import Tensor, SparseTensor
from tensorflow._api.v2 import sparse as tf_sparse
from tensorflow.python.ops.check_ops import assert_equal
from tensorflow.python.ops.sparse_ops import from_dense

from tensor_utils import expand_tensor_leftwards_to_rank_of_tensor, expand_batched_tensor_leftwards_to_rank_of_tensor


def sparse_square_diag(tensor: SparseTensor) -> SparseTensor:
    """
    For a SparseTensor of any rank, this function returns a diagonal matrix based on the innermost dimension.
    I.e., for a SpareTensor with shape [..., a], it will return [..., a, a].
    :param tensor:
    :return:
    """
    with tf.name_scope('sparse_square_diag'):
        # Compute the number of elements in the diagonal.
        n_diag_elems = tensor.shape[-1]

        # Stack coords and transpose.
        coords = tensor.indices[:, -1:]
        other_coords = tensor.indices[:, :-1]
        indices = tf.concat([other_coords, coords, coords], axis=-1)

        # Get the values.
        values = tensor.values

        # Make the shape.
        shape = tensor.shape + [n_diag_elems]

        # Make the diagonal tensor.
        diag_tensor = SparseTensor(indices, values, shape)

        return diag_tensor


def sparse_flatten_at_dim(tensor: SparseTensor, axis=0) -> SparseTensor:
    rank = len(tensor.shape)
    axis = rank + axis if axis < 0 else axis
    if axis == rank - 1:
        return tensor
    elif axis >= rank:
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
    with tf.name_scope('sparse_squeeze'):
        presqueeze_shape = tf.shape(tensor)
        assert_equal(presqueeze_shape[axis], 1)
        squeeze_to = presqueeze_shape.as_list()
        del squeeze_to[axis]
        squeezed_tensor = tf_sparse.reshape(tensor, squeeze_to)
        return squeezed_tensor


def sparse_dense_batched_matmul(tensor_a: Union[ndarray, Tensor, SparseTensor],
                                tensor_b: Union[ndarray, Tensor, SparseTensor],
                                transpose_sparse=False, transpose_dense=False) -> Union[Tensor, SparseTensor]:
    '''
    Computes the matrix multiplication of a sparse tensor and a dense tensor. There is no order to the first two
    tensor arguments to this function. The function assumes that the zeroeth dimension is the batch dimension.

    It is assumed that the dense tensor has a lower rank than the sparse tensor, that the zeroeth dimension of both
    tensors is the batch dimension and identical, and the dense's inner dimensions are compatible with the sparse
    tensor's inner dimensions. For example:
        dense: [<batch_dim>, b, c], sparse: [<batch_dim>, e, a, b].

    The dense tensor should be able to be braodcast onto the sparse tensor's shape. Its shape should be consistent with
    the inner dimensions of the sparse tensor. For example:
        dense: [<batch_dim>, b, c], sparse: [<batch_dim>, a, b, c].

    The following is not currently allowed:
        dense: [<batch_dim>, 1, a, b], sparse: [<batch_dim>, d, a, b].

    The dense tensor will be expanded and tiled to match the sparse tensor's dimensions.

    :param tensor_a: The first tensor, either SparseTensor or Tensor.
    :param tensor_b: The second tensor, either SparseTensor or Tensor.
    :param transpose_sparse: (Default: True) Whether the sparse tensor needs to be transposed first. If both tensors
    are dense, the first tensor will be transposed or not according to this parameter, behaving like transpose_a.
    :param transpose_dense: (Default: True) Whether the dense tensor needs to be transposed first. If both tensors
    are dense, the first tensor will be transposed or not according to this parameter, behaving like transpose_b.
    :return:
    '''
    # If both tensors are dense, pass them to the library function.
    if _both_tensors_are_dense(tensor_a, tensor_b):
        return tf.matmul(tensor_a, tensor_b, transpose_a=transpose_sparse, transpose_b=transpose_dense)

    # If both tensors are sparse, pass them to the batch matmul.
    elif _both_tensors_are_sparse(tensor_a, tensor_b):
        if _tensors_have_the_same_rank(tensor_a, tensor_b):
            return sparse_batched_matmul(tensor_a, tensor_b, transpose_a=transpose_sparse, transpose_b=transpose_dense)
        else:
            return _uneven_sparse_batched_matmul(tensor_a, tensor_b, transpose_a=transpose_sparse,
                                                 transpose_b=transpose_dense)

    # Otherwise find out which one is dense.
    else:
        sparse_is_on_left = False
        if isinstance(tensor_a, (Tensor, ndarray)) and isinstance(tensor_b, SparseTensor):
            dense, sparse = tensor_a, tensor_b

        # Right one must be dense, etc.
        else:
            sparse_is_on_left = True
            sparse, dense = tensor_a, tensor_b

        with tf.name_scope('sparse_dense_batched_matmul'):
            # Ensure that the dense tensor has at least two dimensions.
            assert len(dense.shape) >= 2
            # TODO Manage the case where the dense tensor is actually a SparseTensor.

            # Create a new variable for the dense tensor.
            rank_diff = len(sparse.shape) - len(dense.shape)

            # Determine the new shape of the dense tensor.
            # TODO Assume that the first dimension is the batch dimension
            unshared_dimensions = sparse.shape[1:1 + rank_diff]
            new_non_batch_shape = tf.concat([tf.shape(sparse)[0:1], unshared_dimensions, dense.shape[-2:]], axis=0)

            # Create a placeholder for the new values.
            new_dense = tf.ones(new_non_batch_shape, dtype=dense.dtype)
            new_dense = from_dense(new_dense)

            # Expanded dense tensor.
            dense = expand_batched_tensor_leftwards_to_rank_of_tensor(dense, new_dense)

            # Tile the dense matrix.
            dense = new_dense * dense

            if transpose_sparse and transpose_dense:
                assert sparse.shape[-2] == dense.shape[-1]
            elif transpose_dense:
                assert sparse.shape[-1] == dense.shape[-1]
            elif transpose_sparse:
                assert sparse.shape[-2] == dense.shape[-2]
            else:
                assert sparse.shape[-1] == dense.shape[-2]

            if sparse_is_on_left:
                return sparse_batched_matmul(sparse, dense, transpose_a=transpose_sparse, transpose_b=transpose_dense)
            else:
                return sparse_batched_matmul(dense, sparse, transpose_a=transpose_dense, transpose_b=transpose_sparse)

def _uneven_sparse_batched_matmul(tensor_a: SparseTensor, tensor_b: SparseTensor,
                                  transpose_a=False, transpose_b=False) -> Union[Tensor, SparseTensor]:
    '''
    Expands the rank of one tensor to the other before calling `sparse_batched_matmul`.
    :param tensor_a:
    :param tensor_b:
    :param transpose_a:
    :param transpose_b:
    :return:
    '''
    # Fix expand the sparse tensor to the size of the other.
    smaller_on_left = False
    if len(tensor_a.shape) < len(tensor_b.shape):
        smaller_on_left = True
        lower_rank_tensor, higher_rank_tensor = tensor_a, tensor_b
    else:
        higher_rank_tensor, lower_rank_tensor = tensor_a, tensor_b

    # Create a new variable for the dense tensor.
    rank_diff = len(higher_rank_tensor.shape) - len(lower_rank_tensor.shape)

    print('lower rank:', lower_rank_tensor.shape)
    print('higher rank:', higher_rank_tensor.shape)

    # Determine the new shape of the dense tensor.
    # TODO Assume that the first dimension is the batch dimension
    unshared_dimensions = higher_rank_tensor.shape[1:rank_diff]
    print('unshared_dimensions: ', unshared_dimensions)
    new_non_batch_shape = unshared_dimensions + lower_rank_tensor.shape
    print('new shape:', new_non_batch_shape)

    # Create a placeholder for the new values.
    equal_rank_tensor = tf.ones(new_non_batch_shape, dtype=lower_rank_tensor.dtype)
    equal_rank_tensor = tf.expand_dims(equal_rank_tensor, axis=0)
    equal_rank_tensor = from_dense(equal_rank_tensor)
    print('add', lower_rank_tensor.shape)
    print(' to', equal_rank_tensor.shape)

    equal_rank_tensor = tf.sparse.add(equal_rank_tensor, lower_rank_tensor)
    print(equal_rank_tensor)

    raise ValueError('Both tensors are sparse. Cannot handle this situation yet.')

def sparse_batched_matmul(tensor_a: SparseTensor, tensor_b: SparseTensor,
                          transpose_a=False, transpose_b=False,
                          return_sparse=False) -> Union[Tensor, SparseTensor]:
    '''
    Performs a matrix multiplication on two given `SparseTensors`. They should have a rank ≥ 2 and their rank should
    be equal. Their outermost dimensions must be identical. Their innermost two dimensions must also have a compatible
    intermediate dimension. For example,
        compatible shapes: a.shape: [<batch_dims>, d, e] and b.shape: [<batch_dims>, e, f] before b is transposed.
    By default, `tensor_b` is assumed not to be transposed; i.e., it will need to be transposed within the function
    before being multiplied with `tensor_a`.
    :param tensor_a: A SparseTensor of rank ≥ 2.
    :param tensor_b: A SparseTensor fo rank ≥ 2.
    :param transpose_a: (Bool, default: True) Whether `tensor_a`'s two outermost dimensions need to be transposed
    before multiplying the two tensors.
    :param transpose_b: (Bool, default: True) Whether `tensor_b`'s two outermost dimensions need to be transposed
    before multiplying the two tensors.
    :param return_sparse: (Bool, default True) Whether the function should return a SparseTensor. This parameter
    determines whether the final `tf.sparse.reduce_sum` returns as a SparseTensor or not. WARNING: Enabling this option
    may leave your model unable to perform back-propagation because there is no registered gradient.
    :return:
    '''
    with tf.name_scope('sparse_batched_matmul'):
        # Ensure that the two tensors are of identical rank.
        tensor_a.shape.assert_same_rank(tensor_b.shape)

        # Ensure that they are compatible.
        tensor_a.shape[:-2].assert_is_compatible_with(tensor_b.shape[:-2])

        # Ensure that their rank is at least two.
        tensor_a.shape.with_rank_at_least(2)

        # Transpose the dimensions as required.
        def transpose_outer_dimensions(tensor):
            rank = len(tensor.shape)

            # Change the dimensions according to the rank.
            perm = list(range(rank - 2)) + [rank - 1, rank - 2]

            # Reshape tensor_b to the new shape.
            return tf.sparse.transpose(tensor, perm)

        # Trying to mimic matmul here, so the operation makes no sense.
        tensor_a = transpose_outer_dimensions(tensor_a) if transpose_a else tensor_a
        tensor_b = transpose_outer_dimensions(tensor_b) if transpose_b else tensor_b

        # Determine the tiling factors. The factor of tensor a will be used to tile tensor b and vice versa.
        a_factor = tensor_a.shape[-2]
        b_factor = tensor_b.shape[-1]

        # Expand the dimensions.
        tensor_a = tf.sparse.expand_dims(tensor_a, axis=-1)
        tensor_b = tf.sparse.expand_dims(tensor_b, axis=-3)

        # Tile the respective dimensions.
        tensor_a = sparse_tile_on_axis(tensor_a, -1, b_factor)
        tensor_b = sparse_tile_on_axis(tensor_b, -3, a_factor)

        # Add the indices of each to the other to allow `map_values` to do its magic.
        tensor_a = tf.cast(tf.sparse.add(tensor_a, sparse_zeros_like(tensor_b, tensor_a.dtype)), tensor_a.dtype)
        tensor_b = tf.cast(tf.sparse.add(tensor_b, sparse_zeros_like(tensor_a, tensor_b.dtype)), tensor_b.dtype)

        # Multiply the two tensors.
        intermediate_result = tf.sparse.map_values(tf.multiply, tensor_a, tensor_b)

        # Then reduce the intermediate dimension.
        result = tf.sparse.reduce_sum(intermediate_result, axis=-2, output_is_sparse=return_sparse)

        return result


def sparse_zeros_like(sparse_tensor: SparseTensor, dtype=tf.int32) -> SparseTensor:
    '''
    Returns a SparseTensor shaped like the given SparseTensor but consists of only zeros.
    :param sparse_tensor: The SparseTensor whose shape should be copied.
    :param dtype: The datatype of the returned SparseTensor.
    :return:
    '''
    with tf.name_scope('sparse_zeros_like'):
        indices = sparse_tensor.indices
        values = tf.zeros_like(sparse_tensor.values)
        shape = tf.cast(tf.shape(sparse_tensor), tf.int64)
        return tf.cast(tf.sparse.SparseTensor(indices, values, shape), dtype)


def sparse_ones_like(sparse_tensor: SparseTensor, dtype=tf.int32) -> SparseTensor:
    '''
    Returns a SparseTensor shaped like the given SparseTensor but consists of only ones.
    :param sparse_tensor: The SparseTensor whose shape should be copied.
    :param dtype: The datatype of the returned SparseTensor.
    :return:
    '''
    with tf.name_scope('sparse_ones_like'):
        indices = sparse_tensor.indices
        values = tf.ones_like(sparse_tensor.values)
        shape = tf.cast(tf.shape(sparse_tensor), tf.int64)
        return tf.cast(tf.sparse.SparseTensor(indices, values, shape), dtype)


def sparse_tile_on_axis(sparse_tensor: SparseTensor, axis, multiple) -> SparseTensor:
    '''
    A tile operation that works on SparseTensors. The argument multiples is supposed to function like tf.tile's
    `multiples` argument. This is much faster than tf.sparse.concat if one is trying to tile a dimension.
    :param sparse_tensor: The SparseTensor to tile.
    :param axis: The axis to tile.
    :param multiple: The multiple by which to tile the given axis.
    :return:
    '''
    rank = len(sparse_tensor.shape)
    axis = rank + axis if axis < 0 else axis

    with tf.name_scope('sparse_tile'):
        if axis > 0:
            # Make the first permutation of the data---bring selected axis to outermost dimension.
            original_perm = tf.range(rank)
            axis_to_outermost_perm = tf.concat(([axis], original_perm[:axis], original_perm[axis + 1:]), axis=0)

            # Make the last permutation of the data---bring selected axis back to its position from outermost dimension.
            axis_to_original_pos_perm = tf.concat((tf.range(1, axis + 1), [0], tf.range(axis + 1, rank)), 0)
            transposed_sparse_tensor = tf.sparse.transpose(sparse_tensor, axis_to_outermost_perm)
        else:
            axis_to_original_pos_perm = tf.range(rank)
            transposed_sparse_tensor = sparse_tensor

        # Add a new dimension and tile that.
        transposed_sparse_tensor = tf.sparse.expand_dims(transposed_sparse_tensor, axis=0)

        # The sparse tensor's indices and values.
        transposed_sparse_tensor = tf.sparse.reorder(transposed_sparse_tensor)
        indices = tf.tile(transposed_sparse_tensor.indices, [multiple, 1])
        values = tf.tile(transposed_sparse_tensor.values, [multiple])
        shape = tf.cast(tf.concat([[multiple], tf.shape(transposed_sparse_tensor)[1:]], axis=0), tf.int64)

        # Scale the values of the first column of indices.
        n_entries = tf.shape(sparse_tensor.indices)[0]
        indices_scale = tf.cast(tf.repeat(tf.range(multiple), n_entries), tf.int64)
        indices_scale = tf.expand_dims(indices_scale, axis=0)
        indices = tf.transpose(tf.tensor_scatter_nd_add(tf.transpose(indices), [[0]], indices_scale))

        # Create the tiled sparse tensor.
        tiled_sparse_tensor = tf.sparse.SparseTensor(indices, values, shape)

        # Then flatten.
        tiled_sparse_tensor = sparse_flatten_at_dim(tiled_sparse_tensor, 0)

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


def _tensors_have_the_same_rank(*tensors: Union[Tensor, SparseTensor]):
    x = tf.cast(True, tf.bool)
    # This long-winded formulation is necessary because TF Graph Construction demands an else branch.
    for t in tensors[1:]:
        if not tf.equal(tf.rank(t), tf.rank(tensors[0])):
            x = tf.logical_and(x, False)
        else:
            x = tf.logical_and(x, True)
    return x


def _make_multiples_tensor(tensor_like, multiple, axis):
    return tf.tensor_scatter_nd_update(tf.ones_like(tensor_like), [[axis]], [multiple])
