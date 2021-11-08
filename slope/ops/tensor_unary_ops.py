from __future__ import annotations
import numpy as np
from slope.core.tensor import Tensor

import slope


def slope_pos(tensor):
    return tensor


def slope_pos_grad(tensor, grad):
    return grad


pos = slope.UnaryOperation(
    slope_pos,
    slope_pos_grad
)


def slope_neg(tensor):
    return np.negative(tensor)


def slope_neg_grad(tensor, grad):
    return np.negative(grad)


neg = slope.UnaryOperation(
    slope_neg,
    slope_neg_grad
)


def slope_abs(tensor):
    return np.abs(tensor)


def slope_abs_grad(tensor, grad):
    mask = tensor < 0

    return np.subtract(np.multiply(~mask, grad), np.multiply(mask, grad))


abs = slope.UnaryOperation(
    slope_abs,
    slope_abs_grad
)


def slope_exp(tensor):
    return np.exp(tensor)


def slope_exp_grad(tensor, grad):
    return np.exp(grad)


exp = slope.UnaryOperation(
    slope_exp,
    slope_exp_grad
)


def slope_log(tensor: Tensor) -> Tensor:
    return np.log(tensor)


def slope_log_grad(tensor: Tensor, grad: Tensor) -> Tensor:
    return np.divide(grad, tensor)


log = slope.UnaryOperation(
    slope_log,
    slope_log_grad
)


def slope_sin(tensor):
    return np.sin(tensor)


def slope_sin_grad(tensor, grad):
    return np.cos(tensor)


sin = slope.UnaryOperation(
    slope_sin,
    slope_sin_grad
)


def slope_cos(tensor):
    return np.cos(tensor)


def slope_cos_grad(tensor, grad):
    return np.negative(np.sin(tensor))


cos = slope.UnaryOperation(
    slope_cos,
    slope_cos_grad
)
