import numpy as np


class Tensor(np.ndarray):
    def __new__(cls, array):
        return np.asarray(array).view(cls)

    def grad(self, grad):
        return grad
