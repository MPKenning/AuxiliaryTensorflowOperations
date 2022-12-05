from time import time

import numpy as np
import tensorflow as tf
from unittest import TestCase

from numpy import mean, product
from numpy.testing import assert_allclose, assert_equal
from tensorflow.python.ops.sparse_ops import from_dense, sparse_tensor_to_dense as to_dense

from sparse_tensor_utils import sparse_batched_matmul, sparse_dense_batched_matmul, sparse_flatten_at_dim, \
    sparse_square_diag, sparse_tile_on_axis


def _make_dense_and_sparse_masked_matrices(shape):
    dense = np.random.rand(*shape)
    mask = np.random.randint(2, size=shape)
    dense *= mask
    sparse = from_dense(dense)
    sparse.set_shape(shape)
    return dense, sparse


class SparseUtilsTest(TestCase):

    def test_sparse_batched_matmul_with_non_batched_args_returns_correct_result(self):
        shape = (10, 20)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        shape = (20, 50)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(a, b)
        expected2 = tf.sparse.sparse_dense_matmul(a_sparse, b)
        result = to_dense(sparse_batched_matmul(a_sparse, b_sparse)).numpy()

        assert_allclose(expected, expected2)
        assert_allclose(expected, result)

    def test_sparse_batched_matmul_with_batched_args_returns_correct_result(self):
        shape = (2, 4, 10, 20)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        shape = (2, 4, 20, 50)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(a, b)
        result = to_dense(sparse_batched_matmul(a_sparse, b_sparse)).numpy()

        assert_allclose(expected, result)

    def test_sparse_batched_matmul_batched_identically_sized_inputs_and_transpose_of_first_returns_correct_result(
            self):
        shape = (2, 4, 10, 20)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(a, b, transpose_a=True)
        result = to_dense(sparse_batched_matmul(a_sparse, b_sparse, transpose_a=True)).numpy()

        assert_allclose(expected, result)

    def test_sparse_batched_matmul_batched_identically_sized_inputs_and_transpose_of_second_returns_correct_result(
            self):
        shape = (2, 4, 10, 20)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(a, b, transpose_b=True)
        result = to_dense(sparse_batched_matmul(a_sparse, b_sparse, transpose_b=True)).numpy()

        assert_allclose(expected, result)

    def test_sparse_batched_matmul_compiles_with_jit_compiler(self):
        shape = (2, 4, 10, 20)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        @tf.function(jit_compile=True)
        def dense_matmul():
            return tf.matmul(a, b, transpose_b=True)

        @tf.function(jit_compile=True)
        def sparse_matmul():
            return to_dense(sparse_batched_matmul(a_sparse, b_sparse, transpose_b=True))

        expected = dense_matmul()
        result = sparse_matmul()

        assert_allclose(expected, result)

    def test_sparse_matmul_of_rank_4_sparse_tensor_and_rank_2_dense_tensor_returns_correct_result(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        shape = (20, 10)
        b, _ = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(to_dense(a_sparse), b[None, None])
        result = sparse_dense_batched_matmul(a_sparse, tf.cast(b, b.dtype))
        result = to_dense(result)

        assert_allclose(expected, result)

    def test_sparse_matmul_of_rank_2_dense_tensor_and_rank_4_sparse_tensor_returns_correct_result(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        shape = (20, 10)
        b, _ = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(b[None, None], to_dense(a_sparse))
        result = sparse_dense_batched_matmul(tf.cast(b, b.dtype), a_sparse)
        result = to_dense(result)

        assert_allclose(expected, result)

    def test_sparse_matmul_of_rank_3_dense_tensor_and_rank_4_sparse_tensor_returns_correct_result(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        shape = (4, 20, 10)
        b, _ = _make_dense_and_sparse_masked_matrices(shape)

        expected = tf.matmul(to_dense(a_sparse), b[None])
        result = sparse_dense_batched_matmul(a_sparse, tf.cast(b, b.dtype))
        result = to_dense(result)

        assert_allclose(expected, result)

    def test_sparse_flatten_on_first_two_dims_returns_correct_shape(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        result = sparse_flatten_at_dim(a_sparse, 0)

        assert_equal((8, 10, 20), result.shape)

    def test_sparse_flatten_on_last_two_dims_returns_correct_shape(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        result = sparse_flatten_at_dim(a_sparse, 2)

        assert_equal((2, 4, 200), result.shape)

    def test_sparse_flatten_on_minus_two_dim_returns_correct_shape(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        result = sparse_flatten_at_dim(a_sparse, -2)

        assert_equal((2, 4, 200), result.shape)

    def test_sparse_flatten_on_out_of_bounds_throws_error(self):
        shape = (2, 4, 10, 20)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        with self.assertRaises(IndexError):
            sparse_flatten_at_dim(a_sparse, 4)

    def test_for_sparse_vector_sparse_square_diag_returns_square_matrix(self):
        shape = (10,)
        expected_shape_of_result = (10, 10)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        result = sparse_square_diag(a_sparse)

        assert_equal(expected_shape_of_result, result.shape)

    def test_for_sparse_matrix_sparse_square_diag_returns_cubic_tensor(self):
        shape = (10, 10)
        expected_shape_of_result = (10, 10, 10)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        result = sparse_square_diag(a_sparse)

        assert_equal(expected_shape_of_result, result.shape)

    def test_sparse_scales_better_than_dense(self):
        def dense_matmul(in_a, in_b):
            tf.matmul(in_a, in_b)

        def sparse_matmul(in_a, in_b):
            sparse_batched_matmul(in_a, in_b)

        shapes = [(10, 10), (10, 10, 10), (10, 10, 10, 10)]
        repeat = 10
        result_strings = []
        for shape in shapes:
            print(f'Testing rank {len(shape)} matrices with {product(shape)} elements')
            a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
            b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

            dense_time = []
            for i in range(repeat):
                self._start_timer()
                dense_matmul(a, b)
                dense_time += [self._stop_timer_and_record_time('dense', False)]
            dense_time = mean(dense_time)
            print(f'Average dense time on rank {len(shape)}: {dense_time}')

            sparse_time = []
            for i in range(repeat):
                self._start_timer()
                sparse_matmul(a_sparse, b_sparse)
                sparse_time += [self._stop_timer_and_record_time('sparse', False)]
            sparse_time = mean(sparse_time)
            print(f'Average sparse time on rank {len(shape)}: {sparse_time}')

            result_strings += [f'Sparse is better on rank {len(shape)}: {dense_time >= sparse_time}']

        for i in result_strings:
            print(i)

    def test_sparse_scales_better_than_dense_with_graph_execution(self):
        @tf.function
        def dense_matmul(in_a, in_b):
            tf.matmul(in_a, in_b)

        @tf.function
        def sparse_matmul(in_a, in_b):
            sparse_batched_matmul(in_a, in_b)

        shapes = [(10, 10), (10, 10, 10), (10, 10, 10, 10)]
        repeat = 10
        result_strings = []
        for shape in shapes:
            print(f'Testing rank {len(shape)} matrices with {product(shape)} elements')
            a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
            b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

            dense_time = []
            for i in range(repeat):
                self._start_timer()
                dense_matmul(a, b)
                dense_time += [self._stop_timer_and_record_time('dense', False)]
            dense_time = mean(dense_time)
            print(f'Average dense time on rank {len(shape)}: {dense_time}')

            sparse_time = []
            for i in range(repeat):
                self._start_timer()
                sparse_matmul(a_sparse, b_sparse)
                sparse_time += [self._stop_timer_and_record_time('sparse', False)]
            sparse_time = mean(sparse_time)
            print(f'Average sparse time on rank {len(shape)}: {sparse_time}')

            result_strings += [f'Sparse is better on rank {len(shape)}: {dense_time >= sparse_time}']

        for i in result_strings:
            print(i)

    def test_sparse_matmul_is_better_than_looped_matmul(self):
        def loop_sparse_matmul(a, b):
            first_dim = tf.shape(a)[0]
            second_dim = tf.shape(a)[1]
            third_dim_a = tf.shape(a)[2]
            fourth_dim_a = tf.shape(a)[3]
            third_dim_b = tf.shape(b)[2]
            fourth_dim_b = tf.shape(b)[3]

            outputs = tf.zeros((first_dim, second_dim, third_dim_a, fourth_dim_b), dtype=a.dtype)
            for j in range(first_dim):
                for i in range(second_dim):
                    a_slice = tf.sparse.slice(a, [j, i, 0, 0], [1, 1, third_dim_a, fourth_dim_a])
                    a_slice = tf.sparse.reshape(a_slice, (third_dim_a, fourth_dim_a))

                    b_slice = tf.slice(b, [j, i, 0, 0], [1, 1, third_dim_b, fourth_dim_b])
                    b_slice = tf.reshape(b_slice, (third_dim_b, fourth_dim_b))

                    partial_outputs = tf.sparse.sparse_dense_matmul(a_slice, b_slice)
                    partial_outputs = tf.expand_dims(partial_outputs, axis=0)

                    outputs = tf.tensor_scatter_nd_add(outputs, [[j, i]], partial_outputs)

            return outputs

        def sparse_matmul(a, b):
            return sparse_batched_matmul(a, b)

        shape = (64, 16, 5, 10)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        shape = (64, 16, 10, 20)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        repeat = 10
        loop_sparse_time = []
        outputs_loop_sparse = None
        for i in range(repeat):
            self._start_timer()
            outputs_loop_sparse = loop_sparse_matmul(a_sparse, b)
            loop_sparse_time += [self._stop_timer_and_record_time('sparse', False)]
        loop_sparse_time = mean(loop_sparse_time)
        print(f'Average loop sparse time: {loop_sparse_time}')

        sparse_time = []
        outputs_sparse = None
        for i in range(repeat):
            self._start_timer()
            outputs_sparse = sparse_matmul(a_sparse, b_sparse)
            sparse_time += [self._stop_timer_and_record_time('sparse', False)]
        sparse_time = mean(sparse_time)
        outputs_sparse = to_dense(outputs_sparse)
        print(f'Average sparse time: {sparse_time}')

        print(f'Sparse is better on rank {len(shape)}: {loop_sparse_time >= sparse_time}')

        assert_allclose(outputs_loop_sparse, outputs_sparse)

    def test_sparse_matmul_is_better_than_looped_matmul_on_graph_execution(self):
        @tf.function
        def dense_matmul(in_a, in_b):
            return tf.matmul(in_a, in_b)

        @tf.function
        def loop_sparse_matmul(a, b):
            first_dim = tf.shape(a)[0]
            second_dim = tf.shape(a)[1]
            third_dim_a = tf.shape(a)[2]
            fourth_dim_a = tf.shape(a)[3]
            third_dim_b = tf.shape(b)[2]
            fourth_dim_b = tf.shape(b)[3]

            outputs = tf.zeros((first_dim, second_dim, third_dim_a, fourth_dim_b), dtype=a.dtype)
            for j in range(first_dim):
                for i in range(second_dim):
                    a_slice = tf.sparse.slice(a, [j, i, 0, 0], [1, 1, third_dim_a, fourth_dim_a])
                    a_slice = tf.sparse.reshape(a_slice, (third_dim_a, fourth_dim_a))

                    b_slice = tf.slice(b, [j, i, 0, 0], [1, 1, third_dim_b, fourth_dim_b])
                    b_slice = tf.reshape(b_slice, (third_dim_b, fourth_dim_b))

                    partial_outputs = tf.sparse.sparse_dense_matmul(a_slice, b_slice)
                    partial_outputs = tf.expand_dims(partial_outputs, axis=0)

                    outputs = tf.tensor_scatter_nd_add(outputs, [[j, i]], partial_outputs)

            return outputs

        @tf.function
        def sparse_matmul(a, b):
            return sparse_batched_matmul(a, b)

        shape = (2, 4, 2, 2)
        a, a_sparse = _make_dense_and_sparse_masked_matrices(shape)
        shape = (2, 4, 2, 2)
        b, b_sparse = _make_dense_and_sparse_masked_matrices(shape)

        repeat = 10
        dense_time = []
        dense_result = None
        for i in range(repeat):
            self._start_timer()
            dense_result = dense_matmul(a, b)
            dense_time += [self._stop_timer_and_record_time('dense', False)]
        dense_time = mean(dense_time)
        print(f'Average dense time on rank {len(shape)}: {dense_time}')

        loop_sparse_time = []
        outputs_loop_sparse = None
        for i in range(repeat):
            self._start_timer()
            outputs_loop_sparse = loop_sparse_matmul(a_sparse, b)
            loop_sparse_time += [self._stop_timer_and_record_time('looped_sparse', False)]
        loop_sparse_time = mean(loop_sparse_time)
        print(f'Average loop sparse time: {loop_sparse_time}')

        sparse_time = []
        outputs_sparse = None
        for i in range(repeat):
            self._start_timer()
            outputs_sparse = sparse_matmul(a_sparse, b_sparse)
            sparse_time += [self._stop_timer_and_record_time('sparse', False)]
        sparse_time = mean(sparse_time)
        outputs_sparse = to_dense(outputs_sparse)
        print(f'Average sparse time: {sparse_time}')

        print(f'Sparse is better on rank {len(shape)}: {loop_sparse_time >= sparse_time}')

        assert_allclose(dense_result, outputs_loop_sparse)
        assert_allclose(dense_result, outputs_sparse)
        assert_allclose(outputs_loop_sparse, outputs_sparse)

    def test_sparse_concat_vs_dense_tile(self):
        shape = (2, 2)
        _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

        tile = 10

        print("Sparse concat")
        self._start_timer()
        expected = to_dense(tf.sparse.concat(0, [a_sparse] * 10))
        self._stop_timer_and_record_time('sparse_concat')

        print("Sparse tile")
        self._start_timer()
        result = to_dense(sparse_tile_on_axis(a_sparse, 0, tile))
        self._stop_timer_and_record_time('sparse_tile')

        assert_allclose(expected, result)

    def test_sparse_concat_vs_dense_tile_on_graph_execution(self):
        @tf.function
        def sparse_concat(in_a, tile):
            return tf.sparse.concat(0, [in_a] * tile)

        @tf.function
        def sparse_tile(in_a, tile):
            return sparse_tile_on_axis(in_a, 0, tile)

        tile = 10
        shapes = [(10, 10), (10, 10, 10), (10, 10, 10, 10)]
        repeat = 10
        result_strings = []
        for shape in shapes:
            _, a_sparse = _make_dense_and_sparse_masked_matrices(shape)

            print("Sparse concat")
            expected = None
            sparse_concat_time = []
            for i in range(repeat):
                self._start_timer()
                expected = to_dense(sparse_concat(a_sparse, tile))
                sparse_concat_time += [self._stop_timer_and_record_time('sparse_concat', False)]
            sparse_concat_time = mean(sparse_concat_time)
            print(f'Average sparse concat time on rank {len(shape)}: {sparse_concat_time}')

            print("Sparse tile")
            result = None
            sparse_tile_time = []
            for i in range(repeat):
                self._start_timer()
                result = to_dense(sparse_tile(a_sparse, tile))
                sparse_tile_time += [self._stop_timer_and_record_time('sparse_tile', False)]
            sparse_tile_time = mean(sparse_tile_time)
            print(f'Average sparse tile time on rank {len(shape)}: {sparse_tile_time}')

            assert_allclose(expected, result)

            result_strings += [f'Sparse is better on rank {len(shape)}: {sparse_concat_time >= sparse_tile_time}']

        for i in result_strings:
            print(i)

    def _start_timer(self):
        self._start = time()

    def _stop_timer_and_record_time(self, model, printResult=True):
        end = time()
        execution_time = end - self._start
        if printResult:
            print(f'Execution time of {model}: {execution_time}')
        self._start = 0
        return execution_time
