from dataclasses import dataclass
from slope.core import tensor
from slope.core.tensor import Tensor
import numpy as np


@dataclass
class UnaryOperationContext:
    tensor: Tensor


class UnaryOperation:
    def __init__(self, tensor: Tensor) -> None:
        self.ctx = UnaryOperationContext(tensor)


class Neg(UnaryOperation, Tensor):
    def __new__(cls, tensor: Tensor) -> Tensor:
        return Tensor.__new__(cls, -tensor)

    def __init__(self, tensor: Tensor) -> None:
        super().__init__(tensor)

    def grad(self, grad):
        tensor = self.ctx.tensor

        return tensor.grad(-grad)


class Abs(UnaryOperation, Tensor):
    def __new__(cls, tensor: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.abs(tensor))

    def __init__(self, tensor: Tensor) -> None:
        super().__init__(tensor)

    def grad(self, grad):
        tensor = self.ctx.tensor

        grad = grad.copy()
        mask = tensor < 0
        grad[mask] = -grad[mask]

        return tensor.grad(-grad)


class Exp(UnaryOperation, Tensor):
    def __new__(cls, tensor: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.exp(tensor))

    def __init__(self, tensor: Tensor) -> None:
        super().__init__(tensor)

    def grad(self, grad):
        tensor = self.ctx.tensor

        return tensor.grad(np.exp(grad))
