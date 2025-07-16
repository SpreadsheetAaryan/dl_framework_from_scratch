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
        
          grad_for_self = out.grad
          for i, dim_out in enumerate(out.data.shape):
              if i >= len(self.data.shape) or self.data.shape[i] == 1 and dim_out > 1:
                  grad_for_self = np.sum(grad_for_self, axis=i, keepdims=True)
          grad_for_self = np.reshape(grad_for_self, self.data.shape)

          grad_for_other = out.grad
          for i, dim_out in enumerate(out.data.shape):
              if i >= len(other.data.shape) or other.data.shape[i] == 1 and dim_out > 1:
                  grad_for_other = np.sum(grad_for_other, axis=i, keepdims=True)
          grad_for_other = np.reshape(grad_for_other, other.data.shape)

          if self.requires_grad:
              self.grad += grad_for_self
          if other.requires_grad:
              other.grad += grad_for_other

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
        
          grad_for_self = other.data * out.grad
          for i, dim_out in enumerate(out.data.shape):
              if i >= len(self.data.shape) or self.data.shape[i] == 1 and dim_out > 1:
                  grad_for_self = np.sum(grad_for_self, axis=i, keepdims=True)
          grad_for_self = np.reshape(grad_for_self, self.data.shape)

          grad_for_other = self.data * out.grad
          for i, dim_out in enumerate(out.data.shape):
              if i >= len(other.data.shape) or other.data.shape[i] == 1 and dim_out > 1:
                  grad_for_other = np.sum(grad_for_other, axis=i, keepdims=True)
          grad_for_other = np.reshape(grad_for_other, other.data.shape)

          if self.requires_grad:
              self.grad += grad_for_self
          if other.requires_grad:
              other.grad += grad_for_other

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

  def sum(self, axis=None, keepdims=False):
    requires_grad = self.requires_grad
    out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, (self,), 'sum', requires_grad=requires_grad)

    if requires_grad:
        def _backward():
            if self.requires_grad:
                grad_to_add = out.grad
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        grad_to_add = np.expand_dims(grad_to_add, axis=axis)
                    elif isinstance(axis, tuple):
                        for ax in sorted(axis):
                            grad_to_add = np.expand_dims(grad_to_add, axis=ax)
                self.grad += np.broadcast_to(grad_to_add, self.data.shape)
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

  def ln(self):
    requires_grad = self.requires_grad
    out = Tensor(np.log(self.data), (self, ), 'ln', requires_grad=requires_grad)

    if requires_grad:
      def _backward():
        if self.requires_grad:
          self.grad += (1 / self.data) * out.grad

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
