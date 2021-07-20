import slope
from slope.core.tensor import Tensor
import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.W = slope.tensor(np.random.uniform(0, 1, (in_features, out_features)))
        self.b = slope.tensor(np.random.uniform(0, 1, (out_features,)))

    def __call__(self, tensor:  Tensor):
        return slope.add(slope.matmul(tensor, self.W), self.b)
