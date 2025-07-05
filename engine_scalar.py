import math
import numpy as np
import matplotlib.pyplot as plt
import random

class Value:
  """
  Represents a single scalar value in a computational graph, tracking data,
  gradient, and lineage for automatic differentiation.
  """
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0 # Stores the gradient of the final loss w.r.t. this value
    self._prev = set(_children) # Parent nodes in the computation graph
    self._op = _op  # Operation that created this node (e.g., '+', '*')

    # This function holds the local derivative rule for backpropagation.
    # It gets set by each operation (__add__, __mul__, etc.).
    self._backward = lambda: None
    self.label = label # Optional label for graph visualization

  def __repr__(self):
    return f'Value(data={self.data})'

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      # Gradients for addition are 1.0; accumulate from downstream
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      # Gradients for multiplication: apply chain rule, accumulate
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
      # Power rule derivative: n * x^(n-1) * upstream_grad
      self.grad += other * (self.data**(other - 1)) * out.grad
    out._backward = _backward
    return out

  def __truediv__(self, other):
    return self * other**-1

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def tanh(self):
    t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      # Derivative of tanh(x) is 1 - tanh^2(x)
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value((self.data if self.data >= 0 else 0), (self, ), 'ReLU')

    def _backward():
      # Derivative of ReLU(x) is 1 for positive, 0 for non-positive
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def exp(self):
    out = Value(math.exp(self.data), (self, ), 'exp')

    def _backward():
      # Derivative of e^x is e^x
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def backward(self):
    """
    Performs full backpropagation from this node.
    Builds a topological graph and iterates backward to compute gradients.
    """
    topo = []
    visited = set()

    # Recursively build graph in topological order
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v) # Add node after its children

    build_topo(self) # Start graph build from current node

    self.grad = 1.0 # Initialize gradient of the output node to 1 (dL/dL = 1)
    for node in reversed(topo):
      node._backward() # Call local backward function for each node


class Module:
  """Base class for all neural network modules."""
  def zero_grad(self):
    """Resets the gradients of all learnable parameters to zero."""
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    """Returns a list of all learnable parameters in the module."""
    return []

class Neuron(Module):
  """A single neuron applying a weighted sum and optional ReLU activation."""
  def __init__(self, n_in, non_lin=True):
    # Weights and bias are learnable parameters
    self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
    self.b = Value(random.uniform(-1, 1))
    self.non_lin = non_lin # Controls whether to apply activation

  def __call__(self, x):
    # Calculate weighted sum of inputs plus bias (dot product)
    raw = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    # Apply ReLU if non-linear, otherwise return linear output
    return raw.relu() if self.non_lin else raw

  def parameters(self):
    """Returns the neuron's weights and bias."""
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.non_lin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
  """A layer of multiple neurons."""
  def __init__(self, n_in, n_out, **kwargs):
    # Create 'n_out' neurons, each accepting 'n_in' inputs
    self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

  def __call__(self, x):
    # Pass input 'x' through all neurons in the layer
    outs = [n(x) for n in self.neurons]
    # Return single output if only one neuron, else list of outputs
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    """Collects parameters from all neurons within this layer."""
    params = []
    for neuron in self.neurons:
      params.extend(neuron.parameters())
    return params

  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
  """A Multi-Layer Perceptron (MLP) neural network."""
  def __init__(self, n_in, n_outs):
    # Define layer sizes, starting with input features
    sz = [n_in] + n_outs
    self.layers = []
    # Construct layers, ensuring last layer is linear (no activation)
    for i in range(len(n_outs)):
      non_lin_flag = (i != len(n_outs) - 1)
      self.layers.append(Layer(sz[i], sz[i+1], non_lin=non_lin_flag))

  def __call__(self, x):
    """Performs the forward pass through the entire MLP."""
    for layer in self.layers:
      x = layer(x)
    return x # Final output of the network

  def parameters(self):
    """Collects all learnable parameters from all layers in the MLP."""
    out = [p for layer in self.layers for p in layer.parameters()]
    return out

  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
