from dataclasses import dataclass
from slope.core.tensor import Tensor
from typing import Callable, Tuple, Set, Any
import numpy as np


@dataclass
class BinaryOperationContext:
    tensor_left: Tensor
    tensor_right: Tensor
    grad: Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]


class BinaryOperation:
    def __new__(cls, func: Callable[[Tensor, Tensor], Tensor], grad: Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]) -> Any:
        class Operation(Tensor):
            def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
                return super().__new__(cls, func(tensor_left, tensor_right))

            def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
                self.ctx = BinaryOperationContext(tensor_left, tensor_right, grad)

            def keys(self) -> Set[int]:
                keys_left, keys_right = self.ctx.tensor_left.keys(), self.ctx.tensor_right.keys()

                return keys_left.union(keys_right)

            def grad(self, tensor: Tensor, grad: Tensor = None) -> Tensor:

                if grad is None:
                    grad = Tensor(np.ones(self.shape))

                key = id(tensor)

                grad_left, grad_right = self.ctx.grad(self.ctx.tensor_left, self.ctx.tensor_right, grad)
                keys_left, keys_right = self.ctx.tensor_left.keys(), self.ctx.tensor_right.keys()

                if key in keys_left and key in keys_right:
                    return self.ctx.tensor_left.grad(tensor, grad_left) + self.ctx.tensor_right.grad(tensor, grad_right)
                elif key in keys_left:
                    return self.ctx.tensor_left.grad(tensor, grad_left)
                elif key in keys_right:
                    return self.ctx.tensor_right.grad(tensor, grad_right)
                else:
                    raise Exception('No gradients')

        return Operation


def slope_add(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.add(tensor_left, tensor_right)


def slope_add_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (grad, grad)


add = BinaryOperation(
    slope_add,
    slope_add_grad
)


def slope_sub(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.subtract(tensor_left, tensor_right)


def slope_sub_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (grad, np.negative(grad))


sub = BinaryOperation(
    slope_sub,
    slope_sub_grad
)


def slope_mul(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.multiply(tensor_left, tensor_right)


def slope_mul_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (np.multiply(grad, tensor_right), np.multiply(tensor_left, grad))


mul = BinaryOperation(
    slope_mul,
    slope_mul_grad
)


def slope_div(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.divide(tensor_left, tensor_right)


def slope_div_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.divide(np.multiply(grad, tensor_right), np.power(tensor_right, 2)),
        np.negative(np.divide(np.multiply(tensor_left, grad), np.power(tensor_right, 2)))
    )


div = BinaryOperation(
    slope_div,
    slope_div_grad
)


def slope_matmul(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.matmul(tensor_left, tensor_right)


def slope_matmul_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.matmul(grad, np.swapaxes(tensor_right, -1, -2)),
        np.matmul(np.swapaxes(tensor_left, -1, -2), grad)
    )


matmul = BinaryOperation(
    slope_matmul,
    slope_matmul_grad
)


def slope_pow(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.power(tensor_left, tensor_right)


def slope_pow_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.multiply(np.multiply(tensor_right, np.power(tensor_left, np.subtract(tensor_right, 1))), grad),
        np.multiply(np.multiply(np.log(tensor_left), np.power(tensor_left, tensor_right)), grad)
    )


pow = BinaryOperation(
    slope_pow,
    slope_pow_grad
)


def slope_max(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.maximum(tensor_left, tensor_right)


def slope_max_grad(tensor_left: Tensor, tensor_right: Tensor, grad: Tensor) -> Tensor:
    mask = tensor_left < tensor_right

    return (
        np.multiply(~mask, grad),
        np.multiply(mask, grad)
    )


max = BinaryOperation(
    slope_max,
    slope_max_grad
)
