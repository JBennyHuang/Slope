import slope
import numpy as np
import torch

if __name__ == '__main__':

    # x = slope.Tensor([
    #     [2., 5., 2.],
    #     [-4., -8., 7.]
    # ])
    # y = slope.Tensor([
    #     [6., -8., 1.],
    #     [-2., 7., 3.],
    #     [-5., -8., 5.]
    # ])
    # z = slope.Matmul(x, y)

    # print(z)
    # print(z.grad(np.ones((2, 3))))

    # x = torch.tensor([
    #     [2., 5., 2.],
    #     [-4., -8., 7.]
    # ], requires_grad=True)
    # y = torch.tensor([
    #     [6., -8., 1.],
    #     [-2., 7., 3.],
    #     [-5., -8., 5.]
    # ], requires_grad=True)

    # z = torch.matmul(x, y)

    # z.backward(torch.ones(2, 3))

    # print(z)
    # print(x.grad)
    # print(y.grad)

    # ABS
    # a = torch.tensor([-1., 2., -3.], requires_grad=True)
    # y = torch.abs(a)

    # print(y)
    # y.backward(torch.tensor([1., 1., 1.]))
    # print(a.grad)

    # a = slope.Tensor([-1., 2., -3.])

    # y = slope.Abs(a)

    # print(y)
    # print(y.grad(np.array([1., 1., 1.])))

    # MAX
    # a = torch.tensor([-1., 2., -3.], requires_grad=True)
    # b = torch.tensor(0., requires_grad=True)

    # y = torch.max(a, b)

    # y.backward(torch.tensor([1., 1., 1.]))

    # print(a.grad)
    # print(b.grad)

    # a = slope.Tensor([-1., 2., -3.])
    # b = slope.Tensor(0.)

    # y = slope.Max(a, b)

    # print(y)
    # print(y.grad(np.array([1., 1., 1.])))

    # EXP
    # a = torch.tensor([1., 2., 3.], requires_grad=True)
    # b = torch.tensor(2., requires_grad=True)
    # c = torch.tensor(3., requires_grad=True)
    # d = torch.tensor([3., 2., 1.], requires_grad=True)

    # # y = torch.pow(torch.add(torch.pow(a, b), d), c)
    # y = (a ** b + d) ** c
    # print(y)

    # y.backward(torch.tensor([1., 1., 1.]))

    # print(a.grad)
    # print(b.grad)
    # print(c.grad)
    # print(d.grad)

    # a = slope.Tensor([1., 2., 3.])
    # b = slope.Tensor(2.)
    # c = slope.Tensor(3.)
    # d = slope.Tensor([3, 2, 1])

    # # y = slope.Pow(slope.Add(slope.Pow(a, b), d), c)
    # y = (a ** b + d) ** c

    # print(y)
    # print(y.grad(np.array([1., 1., 1.])))

    # class grad_ctx:
    #     def __init__(self, tensor) -> None:
    #         self.tensor = tensor

    #         key = id(self.tensor)

    #         self.ctx = {}

    #         self.ctx[key] = lambda grad: grad if self.tensor.shape else np.sum(grad)

    #     def __call__(self, key, grad) -> Any:
    #         if key in self.ctx:
    #             return self.ctx[key](grad)
    #         else:
    #             raise Exception('No Gradient')

    #     def keys(self):
    #         return self.ctx.keys()

    # class tensor(np.ndarray):
    #     def __new__(cls, input_array):
    #         return np.asarray(input_array).view(cls)

    #     def __init__(self, input_array) -> None:
    #         self.grad_ctx = grad_ctx(self)

    #     def grad(self, tensor, grad=None):
    #         key = id(tensor)

    #         if grad:
    #             return self.grad_ctx(key, grad)
    #         else:
    #             return self.grad_ctx(key, np.ones(self.shape))

    # class bin_op_grad_ctx:
    #     def __init__(self, l, r, grad) -> None:
    #         self.l = l
    #         self.r = r
    #         self.grad = grad

    #         self.ctx_l = {}

    #         for key in self.l.grad_ctx.keys():
    #             self.ctx_l[key] = self.l.grad_ctx

    #         self.ctx_r = {}

    #         for key in self.r.grad_ctx.keys():
    #             self.ctx_r[key] = self.r.grad_ctx

    #     def __call__(self, key, grad) -> Any:
    #         grad_l, grad_r = self.grad(self.l, self.r, grad)

    #         if key in self.ctx_l and key in self.ctx_r:
    #             return self.ctx_l[key](key, grad_l) + self.ctx_r[key](key, grad_r)
    #         elif key in self.ctx_l:
    #             return self.ctx_l[key](key, grad_l)
    #         elif key in self.ctx_r:
    #             return self.ctx_r[key](key, grad_r)
    #         else:
    #             raise Exception('No gradients')

    #     def keys(self):
    #         return set.union(set(self.ctx_l.keys()), set(self.ctx_r.keys()))

    # class bin_op:
    #     def __new__(cls, func, grad) -> Any:
    #         class op(tensor):
    #             def __new__(cls, l, r):
    #                 return tensor.__new__(cls, func(l, r))

    #             def __init__(self, l, r) -> None:
    #                 self.grad_ctx = bin_op_grad_ctx(l, r, grad)

    #         return op

    # add = bin_op(
    #     lambda l, r: np.add(l, r),
    #     lambda l, r, grad: (grad, grad)
    # )

    # sub = bin_op(
    #     lambda l, r: np.subtract(l, r),
    #     lambda l, r, grad: (grad, -grad)
    # )

    # mul = bin_op(
    #     lambda l, r: np.multiply(l, r),
    #     lambda l, r, grad: (np.multiply(grad, r), np.multiply(l, grad))
    # )

    # matmul = bin_op(
    #     lambda l, r: np.matmul(l, r),
    #     lambda l, r, grad: (
    #         np.matmul(grad, np.swapaxes(r, -1, -2)),
    #         np.matmul(np.swapaxes(l, -1, -2), grad)
    #     )
    # )

    # a = tensor([4, 5, 6, 7])
    # b = tensor([5, 4, 3, 2])

    # s = add(mul(a, b), b)

    # print(s)
    # print(s.grad(a))
    # print(s.grad(b))

    # a = torch.tensor([4., 5., 6., 7.], requires_grad=True)
    # b = torch.tensor([5., 4., 3., 2.], requires_grad=True)
    # s = torch.add(torch.mul(a, b), b)

    # s.backward(torch.tensor([1., 1., 1., 1.]))

    # print(s)
    # print(a.grad)
    # print(b.grad)

    import time

    a = np.random.uniform(0, 1, (1000, 2000))
    b = np.random.uniform(0, 1, (2000, 2000))
    c = np.random.uniform(0, 1, (2000, 1000))

    x1 = slope.tensor(a)
    y1 = slope.tensor(b)
    z1 = slope.tensor(c)
    w1 = slope.matmul(slope.matmul(x1, y1), z1)

    grad_memo = {}

    t1 = time.time()
    x1_grad = w1.grad(x1)
    y1_grad = w1.grad(y1)
    z1_grad = w1.grad(z1)
    t2 = time.time()
    x1_grad = w1.grad(x1, grad_memo=grad_memo)
    y1_grad = w1.grad(y1, grad_memo=grad_memo)
    z1_grad = w1.grad(z1, grad_memo=grad_memo)
    t3 = time.time()

    print(f'without memo: {t2-t1}')
    print(f'with memo: {t3-t2}')

    # class Optimizer:
    #     def __init__(self, *parameters: slope.Tensor):
    #         self.parameters = parameters

    #     def minimize(self, tensor: slope.Tensor):
    #         for parameter in self.parameters:
    #             tensor.
    #             pass

    W = slope.tensor(np.random.uniform(0, 1, (5, 5)))
    b = slope.tensor(np.random.uniform(0, 1, (1, 5)))

    x = slope.tensor([[1, 2, 3, 4, 5]])
    y = slope.add(slope.matmul(x, W), b)
    print(y)

    print(y.grad(W))
    print(y.grad(b))
