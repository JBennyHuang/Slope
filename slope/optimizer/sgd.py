from slope import Optimizer
import numpy as np
from slope import Tensor


class SGD(Optimizer):
    def __init__(self, *parameters, alpha):
        super().__init__(*parameters)
        self.alpha = alpha

    def apply_grad(self):
        for parameter in self.parameters:
            key = id(parameter)

            new_parameter = np.subtract(parameter, np.multiply(self.alpha, self.grad[key]))

            np.copyto(parameter, new_parameter)
