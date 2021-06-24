from slope.core.tensor import Tensor
import slope
import numpy as np

if __name__ == '__main__':
    x = slope.Tensor([1., 3., -6.])
    y = slope.Tensor([8., -5., 2.])
    w = slope.Div(x, y)
    z = slope.Abs(w)

    print(z.eval())
    print(z.grad(np.array([1., 1., 1.])))
    