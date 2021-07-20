from dataclasses import dataclass
from slope.core.tensor import Tensor
from typing import Callable, Tuple, Set, Dict, Union, Any
import numpy as np


@dataclass
class BinaryOperationContext:
    tensor_left: Tensor
    tensor_right: Tensor
    grad: Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]
    keys: Tuple[Set[int], Set[int]]


class BinaryOperation:
    def __new__(cls, func: Callable[[Tensor, Tensor], Tensor], grad: Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]) -> Any:
        class Operation(Tensor):
            def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
                return super().__new__(cls, func(tensor_left, tensor_right))

            def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
                self.ctx = BinaryOperationContext(tensor_left, tensor_right, grad, (tensor_left.keys(), tensor_right.keys()))

            def keys(self) -> Set[int]:
                return set.union(*self.ctx.keys, set([id(self)]))

            # TODO tail recursion optimization
            def grad(self, tensor: Tensor, grad: Tensor = None, grad_memo: Dict[int, Union[Tensor, Tuple[Tensor, ...]]] = None) -> Tensor:

                if grad is None:
                    grad = Tensor(np.ones(self.shape))
                else:
                    try:
                        grad = np.reshape(grad, (-1, ) + self.shape)
                        grad = np.sum(grad, axis=0)
                    except:
                        raise Exception(f'gradient with shape {grad.shape} do not match tensor with shape {self.shape}')

                key = id(tensor)

                if not (grad_memo is None):
                    key_self = id(self)

                    if not (key_self in grad_memo):
                        grad_memo[key_self] = self.ctx.grad(self.ctx.tensor_left, self.ctx.tensor_right, grad)

                    grad_left, grad_right = grad_memo[key_self]
                else:
                    grad_left, grad_right = self.ctx.grad(self.ctx.tensor_left, self.ctx.tensor_right, grad)

                keys_left, keys_right = self.ctx.keys

                if key in keys_left and key in keys_right:
                    return np.add(self.ctx.tensor_left.grad(tensor, grad_left, grad_memo), self.ctx.tensor_right.grad(tensor, grad_right, grad_memo))
                elif key in keys_left:
                    return self.ctx.tensor_left.grad(tensor, grad_left, grad_memo)
                elif key in keys_right:
                    return self.ctx.tensor_right.grad(tensor, grad_right, grad_memo)
                else:
                    # TODO return zero
                    raise Exception(f'no gradient for tensor with id {key}')

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
