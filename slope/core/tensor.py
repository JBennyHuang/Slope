import numpy as np


class Tensor:
    def __init__(self, tensor):
        self.tensor = np.array(tensor)

    def eval(self):
        return self.tensor

    def grad(self, grad):
        return grad
