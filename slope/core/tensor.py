from __future__ import annotations
import numpy as np
import slope
from typing import Set, Dict, Union, Tuple
from abc import abstractmethod


class Tensor(np.ndarray):
    def __new__(cls, input_array) -> Tensor:
        return np.asarray(input_array).view(cls)

    def keys(self) -> Set[int]:
        return set([id(self)])

    # TODO tail recursion optimization
    @abstractmethod
    def grad(self, tensor: Tensor, grad: Tensor = None, grad_memo: Dict[int, Union[Tensor, Tuple[Tensor, ...]]] = None) -> Tensor:
        if grad is None:
            grad = Tensor(np.ones(self.shape))
        else:
            try:
                grad = np.reshape(grad, (-1, ) + self.shape)
                grad = np.sum(grad, axis=0)
            except:
                raise Exception(f'gradient with shape {grad.shape} do not match tensor with shape {self.shape}')

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
