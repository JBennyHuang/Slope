from __future__ import annotations
import numpy as np
from numpy.core.fromnumeric import shape
import slope
from typing import Set


class Tensor(np.ndarray):
    def __new__(cls, input_array) -> Tensor:
        return np.asarray(input_array).view(cls)

    def keys(self) -> Set[int]:
        return set([id(self)])

    def grad(self, tensor: Tensor, grad: Tensor = None) -> Tensor:
        # TODO shape mismatch for broadcasting

        if grad is None:
            grad = Tensor(np.ones(self.shape))

        for _ in range(len(grad.shape) - len(self.shape)):
            grad = np.sum(grad, axis=0)

        for dim in range(len(self.shape)):
            if self.shape[dim] == 1:
                grad = np.sum(grad, axis=dim, keepdims=True)

        return grad

    def __pos__(self):
        return slope.pos(self)

    def __neg__(self):
        return slope.neg(self)

    def __add__(self, value):
        return slope.add(self, value)

    def __sub__(self, value):
        return slope.sub(self, value)

    def __mul__(self, value):
        return slope.mul(self, value)

    def __truediv__(self, value):
        return slope.div(self, value)

    def __pow__(self, value):
        return slope.pow(self, value)


tensor = Tensor
