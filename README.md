[I have shared this code on Codelab, too, with some examples.](https://colab.research.google.com/drive/1BvNLVQFrNv9fGJ2lUogtJHji1D1Y2djB?usp=sharing)

### Auxiliary Tensorflow Operations

This is a repository with some helpful operations for Tensorflow in Python.


### Code to Perform Matrix Multiplications on (n > 2)-rank SparseTensors in TensorFlow

Tensorflow lacks the support to perform matrix multiplications (1) on two sparse tensors that share one or more dimensions in common and (2) on a sparse tensor and a dense tensor that have compatible inner dimensions.

I have made code to fix that. It is not the most efficient way, but it works, and my hope is that the memory footprint of such computations is now smaller.

Please let me know of any optimisations you find and I will gladly credit you somehow. Perhaps I will set up a repository or something.

Contact me: Michael Kenning, m.p.kenning@gmail.com.

#### Multiply two sparse tensors of any rank.

The limit is the computer's memory. The only requirement is that their rank is the same. Suppose we have two SparseTensors with shapes:

> A: `[a, b, c, d, e]` and B: `[a, b, c, e, f]`.

The two sparse tensors may be multiplied with `sparse_batched_matmul`, which would return a new tensor with shape `[a, b, c, d, f]`.

The following shapes are not compatible:

> A: `[a, b, c, d, e]` and B: `[z, b, c, e, f]`.

Even if `a` is a factor of `z` or vice versa, they will not multiply. One would need to tile whichever dimension first using `sparse_tile_on_axis`.

#### Multiply one sparse tensor of any rank with a compatible dense tensor.

The limit is again the computer's memory. The only requirement is that the two tensors' two innermost dimensions are compatible. The rest is broadcasted Suppose we have a SparseTensor A and a Tensor B with shapes:

> A: `[a, b, c, d, e]` and B: `[e, f]`.

The result would be a SparseTensor of shape `[a, b, c, d, f]`. The following two shapes would lead to the same result:

> A: `[a, b, c, d, e]` and B: `[c, e, f]`.

The following is not compatible:

> A: `[a, b, c, d, e]` and B: `[z, e, f]`.