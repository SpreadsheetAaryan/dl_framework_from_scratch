import math
import numpy as np
import matplotlib.pyplot as plt
import random

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = np.zeros_like(p.data)

  def parameters(self):
    return []

class Linear(Module):
  def __init__(self, n_in, n_out, non_lin=True):
    self.w = Tensor((np.random.uniform(low=-1, high=1, size=(n_in, n_out))), requires_grad=True)
    self.b = Tensor((np.random.uniform(low=-1, high=1, size=(1, n_out))), requires_grad=True)
    self.non_lin = non_lin

  def __call__(self, x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    assert np.shape(x.data)[1] == np.shape(self.w.data)[0], "data has wrong shape"
    out = x @ self.w + self.b
    act = out.relu() if self.non_lin else out
    return act

  def parameters(self):
    return [self.w, self.b]

  def __repr__(self):
    return f"{'ReLU' if self.non_lin else 'Linear'}Layer({self.w.data.shape[0]}, {self.w.data.shape[1]})"

class MLP_Tensor(Module):
  def __init__(self, n_in, n_outs):
    sz = [n_in] + n_outs
    self.layers = []

    for i in range(len(n_outs)):
      non_lin_flag = (i != len(n_outs) - 1)
      self.layers.append(Layer(sz[i], sz[i+1], non_lin=non_lin_flag))

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    out = [p for layer in self.layers for p in layer.parameters()]
    return out

  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
