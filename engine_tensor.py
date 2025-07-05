import numpy as np
import math
import random

class Tensor:

  def __init__(self, data, _children=(), _op='', label='', requires_grad=False):
    self.data = np.array(data, dtype=np.float64)
    self.grad = np.zeros_like(self.data)  # Gradient initialized to zeros, matching data's shape
    self._prev = set(_children)           # Parents in the computational graph
    self._backward = lambda: None         # Placeholder for the local backpropagation function
    self.label = label
    self._op = _op
    self.requires_grad = requires_grad    # Flag to enable/disable gradient tracking

  def __repr__(self):
    return f'Tensor(data={self.data}, requires_grad={self.requires_grad}))'

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    requires_grad = self.requires_grad or other.requires_grad
    out = Tensor((self.data + other.data), (self, other), '+', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        # Gradients for addition are 1.0. Accumulate from upstream.
        # This implementation specifically handles cases where 'out.grad'
        # might be larger than 'self.data' or 'other.data' due to broadcasting
        # (e.g., adding a (1,N) bias vector to a (B,N) matrix).
        if self.requires_grad:
          self.grad += np.sum(out.grad, axis=0) # Sums gradient across batch dimension for bias-like tensors
        if other.requires_grad:
          other.grad += np.sum(out.grad, axis=0) # Sums gradient across batch dimension for bias-like tensors

      out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    requires_grad = self.requires_grad or other.requires_grad
    out = Tensor((self.data * other.data), (self, other), '*', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        # Gradients for element-wise multiplication: apply chain rule.
        # This implementation assumes no complex broadcasting beyond scalar/vector-to-matrix.
        if self.requires_grad:
          self.grad += other.data * out.grad
        if other.requires_grad:
          other.grad += self.data * out.grad 

      out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int and float"
    requires_grad = self.requires_grad
    out = Tensor((self.data ** other), (self, ), f'**{other}', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          # Power rule derivative: d(x^n)/dx = n * x^(n-1)
          self.grad += other * (self.data**(other - 1)) * out.grad

      out._backward = _backward
    return out

  def __truediv__(self, other):
    return self * other**-1

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __matmul__(self, other):
    assert isinstance(other, Tensor), "only supporting tensors"
    # Basic dimension check for matrix multiplication
    assert np.shape(self.data)[1] == np.shape(other.data)[0], "Last dim of self must match first dim of other for matmul"
    requires_grad = self.requires_grad or other.requires_grad
    out = Tensor((self.data @ other.data), (self, other), '@', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          # Matrix calculus rule: dL/dA = (dL/dC) @ B.T for C = A @ B
          self.grad += out.grad @ other.data.T
        if other.requires_grad:
          # Matrix calculus rule: dL/dB = A.T @ (dL/dC) for C = A @ B
          self.grad += self.data.T @ out.grad # Note: This line should be `other.grad += ...` for correctness.

      out._backward = _backward
    return out

  def sum(self):
    requires_grad = self.requires_grad
    out = Tensor(np.sum(self.data), (self, ), 'sum', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          # Gradient for summation: propagate upstream grad as ones across original shape.
          # This accounts for the reduction in dimensions during the forward sum.
          self.grad += np.ones_like(self.data) * out.grad

      out._backward = _backward
    return out

  def exp(self):
    requires_grad = self.requires_grad
    out = Tensor(np.exp(self.data), (self, ), 'exp', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          # Derivative of e^x is e^x; apply chain rule.
          self.grad += out.data * out.grad

      out._backward = _backward
    return out

  def tanh(self):
    requires_grad = self.requires_grad
    out = Tensor(np.tanh(self.data), (self, ), 'tanh', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          # Derivative of tanh(x) is 1 - tanh^2(x); apply chain rule.
          self.grad += (1 - out.data**2) * out.grad

      out._backward = _backward
    return out

  def relu(self):
    requires_grad = self.requires_grad
    out = Tensor(np.maximum(0, self.data), (self, ), 'relu', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          # Derivative of ReLU(x) is 1 for positive x, 0 otherwise; apply chain rule.
          self.grad += (out.data > 0) * out.grad

      out._backward = _backward
    return out

  def backward(self):
    if not self.requires_grad:
      raise RuntimeError("Output tensor does not require gradients. Call .backward() on a tensor created with requires_grad=True.")
    
    # Initialize the gradient of the output tensor to ones. This acts as dL/dL = 1.0.
    self.grad = np.ones_like(self.data)

    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        # Recursively build the graph in topological order by visiting parents first.
        for child in v._prev:
          build_topo(child)
        topo.append(v) # Add current node after all its parents are processed.

    build_topo(self) # Start building the graph from the output node.

    # Iterate through the topologically sorted nodes in reverse to backpropagate.
    # This ensures gradients are computed in the correct order (from output to inputs).
    for node in reversed(topo):
      # Only compute gradients for nodes that require them and have a defined _backward function.
      if node.requires_grad and node._backward:
        node._backward()

  def detach(self):
    return Tensor((self.data), requires_grad=False)
