import slope
from slope import Tensor
from abc import abstractmethod


class Optimizer:
    def __init__(self, *parameters: Tensor):
        self.parameters = parameters

        self.grad_map = {}

        self.reset_grad()

    def collect_grad(self, tensor: Tensor):
        grad_memo = {}

        for parameter in self.parameters:
            key = id(parameter)

            grad = parameter.grad(tensor, grad_memo=grad_memo)

            self.grad_map[key] = np.add(self.grad_map[key], grad)

    def reset_grad(self):
        for parameter in self.parameters:
            key = id(parameter)

            self.grad_map[key] = slope.tensor(np.zeros(parameter.shape))

    @abstractmethod
    def apply_grad(self):
        pass
