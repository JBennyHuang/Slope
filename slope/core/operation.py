from typing import Callable, Set, Any, Tuple
import numpy as np

from slope.core.tensor import Tensor


class UnaryOperation:
    def __new__(cls, func: Callable[[Tensor, Tensor], Tensor],
                grad_fn: Callable[[Tensor, Tensor], Tensor]) -> Any:
        class Operation(Tensor):
            def __new__(cls, tensor: Tensor) -> Tensor:
                return super().__new__(cls, func(tensor))

            def __init__(self, tensor: Tensor) -> None:
                self.tensor: Tensor = tensor
                self.grad_fn: Callable[[Tensor, Tensor], Tensor] = grad_fn
                self.keys: Set[int] = tensor.get_keys()

            def get_keys(self) -> Set[int]:
                """get keys of tensors that compose this tensor

                Returns:
                    Set[int]: set of keys of tensors that compose this tensor
                """
                return set.union(self.keys, set([id(self)]))

            def grad(self, tensor: Tensor, grad: Tensor = None) -> Tensor:
                """compute the gradients for this tensor with respect to the tensor

                Args:
                    tensor (Tensor): tensor to compute gradients with respect
                        to
                    grad (Tensor, optional): provided gradients. Defaults to
                        None.

                Returns:
                    Tensor: gradients for this tensor with respect to the
                        tensor
                """
                if grad is None:
                    # default to ones if no grad is provided
                    grad = Tensor(np.ones(self.shape))
                else:
                    # try to broadcast grad to self
                    grad = np.reshape(grad, (-1, ) + self.shape)
                    grad = np.sum(grad, axis=0)

                grad = self.grad_fn(self.tensor, grad)

                target_key = id(tensor)
                keys = self.keys

                if target_key in keys:
                    return self.tensor.grad(tensor, grad=grad)
                else:
                    return Tensor(np.zeros_like(tensor))

        return Operation


class BinaryOperation:
    def __new__(cls, func: Callable[[Tensor, Tensor], Tensor],
                grad_fn: Callable[[Tensor, Tensor, Tensor],
                Tuple[Tensor, Tensor]]) -> Any:
        class Operation(Tensor):
            def __new__(cls, tensor_left: Tensor,
                        tensor_right: Tensor) -> Tensor:
                return super().__new__(cls, func(tensor_left, tensor_right))

            def __init__(self, tensor_left: Tensor,
                         tensor_right: Tensor) -> None:
                self.tensor_left: Tensor = tensor_left
                self.tensor_right: Tensor = tensor_right
                self.grad_fn: Callable[[Tensor, Tensor, Tensor],
                                       Tuple[Tensor, Tensor]] = grad_fn
                self.keys_left: Set[int] = tensor_left.get_keys()
                self.keys_right: Set[int] = tensor_right.get_keys()

            def get_keys(self) -> Set[int]:
                """get keys of tensors that compose this tensor

                Returns:
                    Set[int]: set of keys of tensors that compose this tensor
                """
                return set.union(self.keys_left, self.keys_right,
                                 set([id(self)]))

            def grad(self, tensor: Tensor, grad: Tensor = None) -> Tensor:
                """compute the gradients for this tensor with respect to the tensor

                Args:
                    tensor (Tensor): tensor to compute gradients with respect
                        to
                    grad (Tensor, optional): provided gradients. Defaults to
                        None.

                Returns:
                    Tensor: gradients for this tensor with respect to the
                        tensor
                """
                if grad is None:
                    # default to ones if no grad is provided
                    grad = Tensor(np.ones(self.shape))
                else:
                    # try to broadcast grad to self
                    grad = np.reshape(grad, (-1, ) + self.shape)
                    grad = np.sum(grad, axis=0)

                grad_left, grad_right = self.grad_fn(self.tensor_left,
                                                     self.tensor_right, grad)

                target_key = id(tensor)
                keys_left, keys_right = self.keys_left, self.keys_right

                if target_key in keys_left and target_key in keys_right:
                    return self.tensor_left.grad(tensor, grad_left) + \
                        self.tensor_right.grad(tensor, grad_right)
                elif target_key in keys_left:
                    return self.tensor_left.grad(tensor, grad_left)
                elif target_key in keys_right:
                    return self.tensor_right.grad(tensor, grad_right)
                else:
                    return Tensor(np.zeros_like(tensor))

        return Operation


class NaryOperation:
    ...
