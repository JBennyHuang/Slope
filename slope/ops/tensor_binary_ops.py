from dataclasses import dataclass
from slope.core.tensor import Tensor
import numpy as np


@dataclass
class BinaryOperationContext:
    tensor_left: Tensor
    tensor_right: Tensor


class BinaryOperation:
    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        self.ctx = BinaryOperationContext(tensor_left, tensor_right)


class Add(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.add(tensor_left, tensor_right))

    def __init__(self, tensor_left, tensor_right) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(grad),
            tensor_right.grad(grad)
        )


class Sub(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.subtract(tensor_left, tensor_right))

    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(grad),
            tensor_right.grad(-grad)
        )


class Mul(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.multiply(tensor_left, tensor_right))

    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(np.multiply(grad, tensor_right)),
            tensor_right.grad(np.multiply(tensor_left, grad))
        )


class Div(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.divide(tensor_left, tensor_right))

    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(np.divide(np.multiply(grad, tensor_right), np.power(tensor_right, 2))),
            tensor_right.grad(-np.divide(np.multiply(tensor_left, grad), np.power(tensor_right, 2)))
        )


class Matmul(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.matmul(tensor_left, tensor_right))

    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(np.matmul(grad, np.swapaxes(tensor_right, -1, -2))),
            tensor_right.grad(np.matmul(np.swapaxes(tensor_left, -1, -2), grad))
        )


class Pow(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.power(tensor_left, tensor_right))

    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(np.multiply(np.multiply(tensor_right, np.power(tensor_left, tensor_right - 1)), grad)),
            tensor_right.grad(np.multiply(np.multiply(np.log(tensor_left), np.power(tensor_left, tensor_right)), grad))
        )


class Max(BinaryOperation, Tensor):
    def __new__(cls, tensor_left: Tensor, tensor_right: Tensor) -> Tensor:
        return Tensor.__new__(cls, np.maximum(tensor_left, tensor_right))

    def __init__(self, tensor_left: Tensor, tensor_right: Tensor) -> None:
        super().__init__(tensor_left, tensor_right)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        mask = tensor_left < tensor_right

        return (
            tensor_left.grad(np.multiply(~mask, grad)),
            tensor_right.grad(np.multiply(mask, grad))
        )
