from dataclasses import dataclass
from slope.core.tensor import Tensor
from typing import Callable, Set, Any
import numpy as np


@dataclass
class UnaryOperationContext:
    tensor: Tensor
    grad: Callable[[Tensor, Tensor], Tensor]


class UnaryOperation:
    def __new__(cls, func: Callable[[Tensor, Tensor], Tensor], grad: Callable[[Tensor, Tensor], Tensor]) -> Any:
        class Operation(Tensor):
            def __new__(cls, tensor: Tensor) -> Tensor:
                return super().__new__(cls, func(tensor))

            def __init__(self, tensor: Tensor) -> None:
                self.ctx = UnaryOperationContext(tensor, grad)

            def keys(self) -> Set[int]:
                keys = self.ctx.tensor.keys()

                return keys

            def grad(self, tensor: Tensor, grad: Tensor = None) -> Tensor:

                if grad is None:
                    grad = Tensor(np.ones(self.shape))

                key = id(tensor)

                grad = self.ctx.grad(self.ctx.tensor, grad)
                keys = self.ctx.tensor.keys()

                if key in keys:
                    return self.ctx.tensor.grad(tensor, grad)
                else:
                    raise Exception('No gradients')

        return Operation


def slope_pos(tensor):
    return tensor


def slope_pos_grad(tensor, grad):
    return grad


pos = UnaryOperation(
    slope_pos,
    slope_pos_grad
)


def slope_neg(tensor):
    return np.negative(tensor)


def slope_neg_grad(tensor, grad):
    return np.negative(grad)


neg = UnaryOperation(
    slope_neg,
    slope_neg_grad
)


def slope_abs(tensor):
    return np.abs(tensor)


def slope_abs_grad(tensor, grad):
    mask = tensor < 0

    return np.subtract(np.multiply(~mask, grad), np.multiply(mask, grad))


abs = UnaryOperation(
    slope_abs,
    slope_abs_grad
)


def slope_exp(tensor):
    return np.exp(tensor)


def slope_exp_grad(tensor, grad):
    return np.exp(grad)


exp = UnaryOperation(
    slope_exp,
    slope_exp_grad
)
