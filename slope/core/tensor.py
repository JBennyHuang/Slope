import numpy as np
import slope


class Tensor(np.ndarray):
    def __new__(cls, array):
        return np.asarray(array).view(cls)

    def grad(self, grad):
        if not self.shape:
            return np.sum(grad)
        else:
            return grad

    def __pos__(self):
        return slope.Pos(self)

    def __neg__(self):
        return slope.Neg(self)

    def __add__(self, value):
        return slope.Add(self, value)

    def __sub__(self, value):
        return slope.Sub(self, value)

    def __mul__(self, value):
        return slope.Mul(self, value)

    def __truediv__(self, value):
        return slope.Div(self, value)

    def __pow__(self, value):
        return slope.Pow(self, value)
