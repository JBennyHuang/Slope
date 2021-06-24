import numpy as np

class TensorUnaryOperation:
    def __init__(self, tensor):
        self.tensor = tensor
        
class Neg(TensorUnaryOperation):
    def __init__(self, tensor):
        super().__init__(tensor)
        
    def eval(self):
        return -self.tensor.eval()
    
    def grad(self, grad):
        return -self.tensor.grad(grad)

class Abs(TensorUnaryOperation):
    def __init__(self, tensor):
        super().__init__(tensor)

    def eval(self):
        return np.abs(self.tensor.eval())
    
    def grad(self, grad):
        mask = self.tensor.eval() < 0
        grad[mask] = -grad[mask]
        return self.tensor.grad(grad) 

class Exp(TensorUnaryOperation):
    def __init__(self, tensor):
        super().__init__(tensor)
        
    def eval(self):
        return np.exp(self.tensor.eval())
        
    def grad(self, grad):
        return np.exp(self.tensor.eval()) * self.tensor.grad(grad)