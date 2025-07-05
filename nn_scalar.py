import math
import numpy as np
import random

class Module:
  
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    return []

class Neuron(Module):
  
  def __init__(self, n_in, non_lin=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
    self.b = Value(random.uniform(-1, 1))
    self.non_lin = non_lin 

  def __call__(self, x):
    raw = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    return raw.relu() if self.non_lin else raw

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.non_lin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
  def __init__(self, n_in, n_out, **kwargs):
    self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    params = []
    for neuron in self.neurons:
      params.extend(neuron.parameters())
    return params

  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
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
