import unittest

import slope
import torch
import numpy as np


class TestAdd(unittest.TestCase):

    def test_add_1(self):
        a = np.random.uniform(0, 1, (5, 5))
        b = np.random.uniform(0, 1, (5, 5))

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.add(x1, y1)

        x1_grad = z1.grad(x1)
        y1_grad = z1.grad(y1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.add(x2, y2)

        z2.backward(torch.ones(5, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))


class TestBroadcasting(unittest.TestCase):

    def test_broadcasting_1(self):
        a = np.random.uniform(0, 1, (5, 5))
        b = np.random.uniform(0, 1)

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.add(x1, y1)

        x1_grad = z1.grad(x1)
        y1_grad = z1.grad(y1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.add(x2, y2)

        z2.backward(torch.ones(5, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))

    def test_broadcasting_2(self):
        a = np.random.uniform(0, 1, (5, 5))
        b = np.random.uniform(0, 1, (5,))

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.add(x1, y1)

        x1_grad = z1.grad(x1)
        y1_grad = z1.grad(y1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.add(x2, y2)

        z2.backward(torch.ones(5, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))

    def test_broadcasting_3(self):
        a = np.random.uniform(0, 1, (5, 5))
        b = np.random.uniform(0, 1, (5, 1))

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.add(x1, y1)

        x1_grad = z1.grad(x1)
        y1_grad = z1.grad(y1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.add(x2, y2)

        z2.backward(torch.ones(5, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))

    def test_broadcasting_4(self):
        a = np.random.uniform(0, 1, (5, 1, 4, 5))
        b = np.random.uniform(0, 1, (1, 6, 1, 5))

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.add(x1, y1)

        x1_grad = z1.grad(x1)
        y1_grad = z1.grad(y1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.add(x2, y2)

        z2.backward(torch.ones(5, 6, 4, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))


class TestMatmul(unittest.TestCase):

    def test_matmul_1(self):

        a = np.random.uniform(0, 1, (5, 5))
        b = np.random.uniform(0, 1, (5, 5))

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.matmul(x1, y1)

        x1_grad = z1.grad(x1)
        y1_grad = z1.grad(y1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.matmul(x2, y2)

        z2.backward(torch.ones(5, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))

    def test_matmul_2(self):
        a = np.random.uniform(0, 1, (5, 10))
        b = np.random.uniform(0, 1, (10, 10))
        c = np.random.uniform(0, 1, (10, 5))

        x1 = slope.tensor(a)
        y1 = slope.tensor(b)
        z1 = slope.tensor(c)
        w1 = slope.matmul(slope.matmul(x1, y1), z1)

        x1_grad = w1.grad(x1)
        y1_grad = w1.grad(y1)
        z1_grad = w1.grad(z1)

        x2 = torch.tensor(a, requires_grad=True)
        y2 = torch.tensor(b, requires_grad=True)
        z2 = torch.tensor(c, requires_grad=True)
        w2 = torch.matmul(torch.matmul(x2, y2), z2)

        w2.backward(torch.ones(5, 5))

        x2_grad = x2.grad
        y2_grad = y2.grad
        z2_grad = z2.grad

        self.assertTrue(np.all(w1.round(5) == w2.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(x1_grad.round(5) == x2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(y1_grad.round(5) == y2_grad.detach().detach().numpy().round(5)))
        self.assertTrue(np.all(z1_grad.round(5) == z2_grad.detach().detach().numpy().round(5)))

    def test_matmul_3(self):
        x = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])

        y = np.array([
            [0.],
            [1.],
            [1.],
            [0.]
        ])

        W1 = np.random.uniform(0, 1, (2, 2))
        b1 = np.random.uniform(0, 1, (1, 2))
        W2 = np.random.uniform(0, 1, (2, 1))
        b2 = np.random.uniform(0, 1, (1, 1))

        x1 = slope.tensor(x)
        y1 = slope.tensor(y)
        W11 = slope.tensor(W1)
        b11 = slope.tensor(b1)
        W21 = slope.tensor(W2)
        b21 = slope.tensor(b2)

        l11 = slope.max(slope.matmul(x1, W11) + b11, slope.tensor(0))
        l21 = slope.matmul(l11, W21) + b21

        z1 = slope.abs(l21 - y1)

        W11_grad = z1.grad(W11)
        b11_grad = z1.grad(b11)
        W21_grad = z1.grad(W21)
        b21_grad = z1.grad(b21)

        x2 = torch.tensor(x)
        y2 = torch.tensor(y)
        W12 = torch.tensor(W1, requires_grad=True)
        b12 = torch.tensor(b1, requires_grad=True)
        W22 = torch.tensor(W2, requires_grad=True)
        b22 = torch.tensor(b2, requires_grad=True)

        l12 = torch.maximum(torch.matmul(x2, W12) + b12, torch.tensor(0))
        l22 = torch.matmul(l12, W22) + b22

        z2 = torch.abs(l22 - y2)

        z2.backward(torch.ones(4, 1))

        W12_grad = W12.grad
        b12_grad = b12.grad
        W22_grad = W22.grad
        b22_grad = b22.grad

        self.assertTrue(np.all(z1.round(5) == z2.detach().numpy().round(5)))
        self.assertTrue(np.all(W11_grad.round(5) == W12_grad.detach().numpy().round(5)))
        self.assertTrue(np.all(b11_grad.round(5) == b12_grad.detach().numpy().round(5)))
        self.assertTrue(np.all(W21_grad.round(5) == W22_grad.detach().numpy().round(5)))
        self.assertTrue(np.all(b21_grad.round(5) == b22_grad.detach().numpy().round(5)))


if __name__ == '__main__':
    unittest.main()
