import numpy as np
import math
import random

class Tensor:

  def __init__(self, data, _children=(), _op='', label='', requires_grad=False):
    self.data = np.array(data, dtype=np.float64)
    self.grad = np.zeros_like(self.data)
    self._prev = set(_children)
    self._backward = lambda: None
    self.label = label
    self._op = _op
    self.requires_grad = requires_grad

  def __repr__(self):
    return f'Tensor(data={self.data}, requires_grad={self.requires_grad}))'

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    requires_grad = self.requires_grad or other.requires_grad
    out = Tensor((self.data + other.data), (self, other), '+', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += np.sum(out.grad, axis=0)
        if other.requires_grad:
          other.grad += np.sum(out.grad, axis=0)

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
    assert np.shape(self.data)[1] == np.shape(other.data)[0], "wrong dimensions"
    requires_grad = self.requires_grad or other.requires_grad
    out = Tensor((self.data @ other.data), (self, other), '@', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += out.grad @ other.data.T
        if other.requires_grad:
          other.grad += self.data.T @ out.grad

      out._backward = _backward
    return out

  def sum(self):
    requires_grad = self.requires_grad
    out = Tensor((np.sum(self.data)), (self, ), 'sum', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += np.ones_like(self.data) * out.grad

      out._backward = _backward
    return out

  def exp(self):
    requires_grad = self.requires_grad
    out = Tensor((np.exp(self.data)), (self, ), 'exp', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += out.data * out.grad

      out._backward = _backward
    return out

  def tanh(self):
    requires_grad = self.requires_grad
    out = Tensor((np.tanh(self.data)), (self, ), 'tanh', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += (1 - np.tanh(self.data)**2) * out.grad

      out._backward = _backward
    return out

  def relu(self):
    requires_grad = self.requires_grad
    out = Tensor((np.maximum(0, self.data)), (self, ), 'relu', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += (out.data > 0) * out.grad

      out._backward = _backward
    return out

  def backward(self):
    if not self.requires_grad:
      raise RuntimeError("Output tensor does not require gradients. Call .backward() on a tensor created with requires_grad=True.")
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
      for child in v._prev:
        build_topo(child)

      topo.append(v)

    build_topo(self)

    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
      if node.requires_grad:
        node._backward()

  def detach(self):
    return Tensor((self.data), requires_grad=False)
