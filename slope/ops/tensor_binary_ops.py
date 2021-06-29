from dataclasses import dataclass

from numpy.lib.type_check import common_type
from slope.core.tensor import Tensor
import numpy as np


@dataclass
class BinaryOperationContext:
    tensor_left: Tensor
    tensor_right: Tensor


class BinaryOperation:
    def __init__(self) -> None:
        self.ctx = None


class Add(BinaryOperation):
    def __init__(self) -> None:
        super().__init__()

    def eval(self, x, y):
        self.ctx = BinaryOperationContext(x, y)
        return np.add(x, y)

    def grad(self, grad):
        tensor_left = self.ctx.tensor_left
        tensor_right = self.ctx.tensor_right

        return (
            tensor_left.grad(grad),
            tensor_right.grad(grad)
        )


# class TensorBinaryOperation:
#     def __init__(self, tensor_left, tensor_right):
#         self.tensor_left = tensor_left
#         self.tensor_right = tensor_right


# class Add(TensorBinaryOperation):
#     def __init__(self, tensor_left, tensor_right):
#         super().__init__(tensor_left, tensor_right)

#     def eval(self):
#         return self.tensor_left.eval() + self.tensor_right.eval()

#     def grad(self, grad):
#         return self.tensor_left.grad(grad), self.tensor_right.grad(grad)


# class Sub(TensorBinaryOperation):
#     def __init__(self, tensor_left, tensor_right):
#         super().__init__(tensor_left, tensor_right)

#     def eval(self):
#         return self.tensor_left.eval() - self.tensor_right.eval()

#     def grad(self, grad):
#         return self.tensor_left.grad(grad), -self.tensor_right.grad(grad)


# class Mul(TensorBinaryOperation):
#     def __init__(self, tensor_left, tensor_right):
#         super().__init__(tensor_left, tensor_right)

#     def eval(self):
#         return self.tensor_left.eval() * self.tensor_right.eval()

#     def grad(self, grad):
#         return (
#             self.tensor_right.eval() * self.tensor_left.grad(grad),
#             self.tensor_left.eval() * self.tensor_right.grad(grad)
#         )


# class Div(TensorBinaryOperation):
#     def __init__(self, tensor_left, tensor_right):
#         super().__init__(tensor_left, tensor_right)

#     def eval(self):
#         return self.tensor_left.eval() / self.tensor_right.eval()

#     def grad(self, grad):
#         return (
#             self.tensor_right.eval() * self.tensor_left.grad(grad) /
#             self.tensor_right.eval() ** 2,
#             -self.tensor_left.eval() * self.tensor_right.grad(grad) /
#             self.tensor_right.eval() ** 2
#         )


# class Matmul(TensorBinaryOperation):
#     def __init__(self, tensor_left, tensor_right):
#         super().__init__(tensor_left, tensor_right)

#     def eval(self):
#         return np.matmul(self.tensor_left.eval(), self.tensor_right.eval())

#     def grad(self, grad):
#         # grad1 = grad @ np.swapaxes(t2, -1, -2)
#         # grad2 = np.swapaxes(t1, -1, -2) @ grad
#         t1 = np.swapaxes(self.tensor_left.eval(), -1, -2)
#         t2 = np.swapaxes(self.tensor_right.eval(), -1, -2)

#         return (
#             self.tensor_left.grad(np.matmul(grad, t2)),
#             self.tensor_right.grad(np.matmul(t1 , grad))
#         )


# class Max(TensorBinaryOperation):
#     def __init__(self, tensor_left, tensor_right):
#         super().__init__(tensor_left, tensor_right)

#     def eval(self):
#         return np.maximum(self.tensor_left.eval(), self.tensor_right.eval())

#     def grad(self, grad):
#         mask = self.tensor_left.eval() < self.tensor_right.eval()
#         return (
#             ~mask * self.tensor_left.grad(grad),
#             mask * self.tensor_right.grad(grad)
#         )
