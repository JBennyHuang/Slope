from __future__ import annotations
import numpy as np
import slope
from typing import Set, Tuple


class Tensor(np.ndarray):
    def __new__(cls, input_array) -> Tensor:
        return np.asarray(input_array).view(cls)

    def get_keys(self) -> Set[int]:
        """get keys of tensors that compose this tensor

        Returns:
            Set[int]: set of keys of tensors that compose this tensor
        """
        return set([id(self)])

    def grad(self, tensor: Tensor, grad: Tensor = None) -> Tensor:
        """compute the gradients for this tensor with respect to the tensor

        Args:
            tensor (Tensor): tensor to compute gradients with respect to
            grad (Tensor, optional): provided gradients. Defaults to None.

        Returns:
            Tensor: gradients for this tensor with respect to the tensor
        """
        if grad is None:
            # default to ones if no gradients are provided
            grad = Tensor(np.ones(self.shape))
        else:
            # try to broadcast grad to self
            grad = np.reshape(grad, (-1, ) + self.shape)
            grad = np.sum(grad, axis=0)

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
