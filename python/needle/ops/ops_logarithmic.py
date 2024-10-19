from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes, int):
            self.axes = tuple([axes])

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = Z.max(axis=self.axes)
        Z_shape = list(Z.shape)
        if self.axes is not None:
            for axis in self.axes:
                Z_shape[axis] = 1
            Z_max_reshaped = Z_max.reshape(tuple(Z_shape))
        else:
            Z_max_reshaped = Z_max.reshape(tuple([1 for _ in Z_shape]))
        Z_normalized = Z - Z_max_reshaped.broadcast_to(Z.shape)
        return array_api.log(array_api.sum(array_api.exp(Z_normalized), axis=self.axes)) + Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z_max = Tensor(Z.numpy().max(axis=self.axes), device=Z.device)

        Z_shape_for_reshape = list(Z.shape)
        if self.axes is not None:
            for axis in self.axes:
                Z_shape_for_reshape[axis] = 1
        else:
            for i in range(len(Z_shape_for_reshape)):
                Z_shape_for_reshape[i] = 1
        Z_shape_for_reshape = tuple(Z_shape_for_reshape)
        Z_shape_for_broadcast = Z.shape

        Z_max_reshaped_broadcasted = broadcast_to(reshape(Z_max, Z_shape_for_reshape), Z_shape_for_broadcast)
        Z_minus_Z_max = Z - Z_max_reshaped_broadcasted
        Z_exp = exp(Z_minus_Z_max)
        Z_sum_exp = broadcast_to(reshape(summation(Z_exp, self.axes), Z_shape_for_reshape), Z_shape_for_broadcast)
        return multiply(broadcast_to(reshape(out_grad, Z_shape_for_reshape), Z_shape_for_broadcast), divide(Z_exp, Z_sum_exp))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

