from __future__ import annotations
from slope.core.tensor import Tensor
import numpy as np

import slope


def slope_add(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.add(tensor_left, tensor_right)


def slope_add_grad(tensor_left: Tensor,
                   ensor_right: Tensor, grad: Tensor) -> Tensor:
    return (grad, grad)


add = slope.BinaryOperation(
    slope_add,
    slope_add_grad
)


def slope_sub(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.subtract(tensor_left, tensor_right)


def slope_sub_grad(tensor_left: Tensor,
                   tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (grad, np.negative(grad))


sub = slope.BinaryOperation(
    slope_sub,
    slope_sub_grad
)


def slope_mul(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.multiply(tensor_left, tensor_right)


def slope_mul_grad(tensor_left: Tensor,
                   tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.multiply(grad, tensor_right),
        np.multiply(tensor_left, grad)
    )


mul = slope.BinaryOperation(
    slope_mul,
    slope_mul_grad
)


def slope_div(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.divide(tensor_left, tensor_right)


def slope_div_grad(tensor_left: Tensor,
                   tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.divide(np.multiply(grad, tensor_right), np.power(tensor_right, 2)),
        np.negative(np.divide(np.multiply(tensor_left, grad),
                    np.power(tensor_right, 2)))
    )


div = slope.BinaryOperation(
    slope_div,
    slope_div_grad
)


def slope_matmul(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.matmul(tensor_left, tensor_right)


def slope_matmul_grad(tensor_left: Tensor,
                      tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.matmul(grad, np.swapaxes(tensor_right, -1, -2)),
        np.matmul(np.swapaxes(tensor_left, -1, -2), grad)
    )


matmul = slope.BinaryOperation(
    slope_matmul,
    slope_matmul_grad
)


def slope_pow(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.power(tensor_left, tensor_right)


def slope_pow_grad(tensor_left: Tensor,
                   tensor_right: Tensor, grad: Tensor) -> Tensor:
    return (
        np.multiply(np.multiply(tensor_right, np.power(
            tensor_left, np.subtract(tensor_right, 1))), grad),
        np.multiply(np.multiply(np.log(tensor_left),
                    np.power(tensor_left, tensor_right)), grad)
    )

    # return (
    #     tensor_right * tensor_left ** (tensor_right - Tensor(1)) * grad,
    #     slope.log(tensor_left) * tensor_left ** tensor_right * grad
    # )


pow = slope.BinaryOperation(
    slope_pow,
    slope_pow_grad
)


def slope_max(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.maximum(tensor_left, tensor_right)


def slope_max_grad(tensor_left: Tensor,
                   tensor_right: Tensor, grad: Tensor) -> Tensor:
    mask = tensor_left < tensor_right

    return (
        np.multiply(~mask, grad),
        np.multiply(mask, grad)
    )


max = slope.BinaryOperation(
    slope_max,
    slope_max_grad
)


def slope_min(tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
    return np.minimum(tensor_left, tensor_right)


def slope_min_grad(tensor_left: Tensor,
                   tensor_right: Tensor, grad: Tensor) -> Tensor:
    mask = tensor_left > tensor_right

    return (
        np.multiply(~mask, grad),
        np.multiply(mask, grad)
    )


min = slope.BinaryOperation(
    slope_min,
    slope_min_grad
)
