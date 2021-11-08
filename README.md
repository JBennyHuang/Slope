
# Slope

Automagic differentiation


## Usage/Examples

```py
  import slope

  W = slope.tensor(np.random.uniform(0, 1, (5, 5)))
  b = slope.tensor(np.random.uniform(0, 1, (1, 5)))

  x = slope.tensor([[1, 2, 3, 4, 5]])
  y = slope.add(slope.matmul(x, W), b) 
  >>> [[6.64067645 8.00542806 7.25178774 6.56762151 5.57035489]]
    
  W_grad = y.grad(W)
  >>> [[1. 1. 1. 1. 1.]
       [2. 2. 2. 2. 2.]
       [3. 3. 3. 3. 3.]
       [4. 4. 4. 4. 4.]
       [5. 5. 5. 5. 5.]]

  b_grad = y.grad(b)
  >>> [[1. 1. 1. 1. 1.]]
```

  
## Running Tests

To run tests, run the following command

```bash
  python3 -m unittest discover tests/
```

  
