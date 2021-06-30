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
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    b = torch.tensor(2., requires_grad=True)
    c = torch.tensor(3., requires_grad=True)
    d = torch.tensor([3., 2., 1.], requires_grad=True)

    # y = torch.pow(torch.add(torch.pow(a, b), d), c)
    y = (a ** b + d) ** c
    print(y)

    y.backward(torch.tensor([1., 1., 1.]))

    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)

    a = slope.Tensor([1., 2., 3.])
    b = slope.Tensor(2.)
    c = slope.Tensor(3.)
    d = slope.Tensor([3, 2, 1])

    # y = slope.Pow(slope.Add(slope.Pow(a, b), d), c)
    y = (a ** b + d) ** c

    print(y)
    print(y.grad(np.array([1., 1., 1.])))

    # x = slope.Tensor([
    #     [0., 0.],
    #     [0., 1.],
    #     [1., 0.],
    #     [1., 1.]
    # ])

    # y = slope.Tensor([
    #     [0.],
    #     [1.],
    #     [1.],
    #     [0.]
    # ])

    # W1 = slope.Tensor(np.random.uniform(0, 1, (2, 2)))
    # b1 = slope.Tensor(np.random.uniform(0, 1, (1, 2)))
    # W2 = slope.Tensor(np.random.uniform(0, 1, (2, 1)))
    # b2 = slope.Tensor(np.random.uniform(0, 1, (1, 1)))

    # l1 = slope.Matmul(x, W1)
    # l2 = slope.Matmul(l1, W2)

    # loss = slope.Sub(l2, y)

    # print(loss)
    # print(loss.grad(np.ones((4, 1))))
