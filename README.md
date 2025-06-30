# Building Autograd from Scratch: A PyTorch-like API in Python

## Description

This project, inspired by Andrej Karpathy's _"Neural Networks: Zero to Hero"_ course, is a deep dive into the fundamental mechanics of modern deep learning frameworks. I've built **micrograd**, a lightweight automatic differentiation engine from scratch in Python.

Think of it as a mini-PyTorch, designed to help you truly grasp how gradients are computed and propagated through a computational graph— the backbone of how neural networks learn.

This hands-on implementation demystifies concepts like the **chain rule**, **backpropagation**, and the structure of **multi-layer perceptrons (MLPs)**, providing a clear view of the operations happening under the hood in libraries like PyTorch and TensorFlow.

## My Learning Journey

My recent deep dive into Andrej Karpathy's 'Neural Networks: Zero to Hero' course truly cemented my understanding of deep learning's core. The cornerstone of this learning was building micrograd, my own automatic differentiation engine. This project wasn't just about coding; it was about truly grasping backpropagation and gradient descent from their mathematical roots. By meticulously creating the Value object – the fundamental unit that tracks both data and gradients – and then carefully implementing operator overloading for arithmetic operations, I directly saw how a computational graph is built, step by step. Integrating a topological sort for the backward() pass then revealed exactly how gradients flow and accumulate through this graph, providing a tangible understanding of the chain rule in action. What truly reinforced this knowledge was taking the framework I had engineered and, on my own initiative, building out a complete machine learning pipeline with it. From model definition to the forward pass, loss calculation, and ultimately, training parameters through my custom backpropagation – this end-to-end process confirmed my deep understanding of the entire learning cycle, revealing what truly operates beneath the surface of high-level libraries like PyTorch.

---

## Features & Components

### 1. The `Value` Object

The core building block of micrograd is the `Value` object, akin to PyTorch's `Tensor` for scalar operations. Each `Value` object encapsulates:

- `data`: The numerical value of the node.
- `grad`: The gradient of the loss with respect to this `Value`, initialized to `0.0`.
- `_prev`: A set of `Value` objects that were inputs to the operation that created this one.
- `_op`: A string representing the operation (`+`, `*`, etc.).
- `_backward()`: A dynamically defined function containing the local derivative calculation specific to the operation that created it, used to propagate gradients to `_prev`.

---

### 2. Operator Overloading

The `Value` class overloads standard Python operators to enable intuitive expressions:

- Addition: `+`, `__add__`, `__radd__`
- Multiplication: `*`, `__mul__`, `__rmul__`
- Exponentiation: `**`, `__pow__`
- Negation: `-`, `__neg__`
- Subtraction: `-`, `__sub__`
- Division: `/`, `__truediv__`
- Functions: `tanh()`, `exp()`

Each operator defines its own `_backward()` function, correctly accumulating gradients even for shared nodes using `+=`.

---

### 3. Topological Sort for Backpropagation

The `backward()` method on a `Value` object:

- Performs a **topological sort** of the computational graph to ensure correct derivative application order.
- Initializes the gradient of the output node to `1.0` (`dL/dL = 1`).
- Iterates in reverse topological order, calling `_backward()` on each node.

---

### 4. Neural Network Modules

Built on top of `Value`, micrograd includes:

- **Neuron**: Computes a weighted sum of inputs plus bias, then applies `tanh`. Exposes weights and bias as parameters.
- **Layer**: A collection of `Neuron`s, each receiving the same inputs and outputting a list of activations.
- **MLP (Multi-Layer Perceptron)**: A stack of `Layer`s forming a feedforward neural network. Manages forward passes and provides a `parameters()` method for accessing all weights and biases.

---

## Installation & Usage

### Clone the repository

```bash
git clone [YOUR_REPO_LINK_HERE]
cd [your-project-directory]
pip install -r requirements.txt
```

### Code Examples
## Basic Value Operations & Backpropagation

```bash
from micrograd.engine import Value

a = Value(-4.0, label='a')
b = Value(2.0, label='b')

# Forward pass
c = a + b; c.label = 'c'
d = a * b + b**3; d.label = 'd'
e = c - d; e.label = 'e'
f = e**2; f.label = 'f'
g = f / 2.0; g.label = 'g'
g += 10.0 / f; g.label = 'g'  

print(f"Final output (g.data): {g.data:.4f}")

# Backward pass
g.backward()

print(f"Gradient of g with respect to a (dg/da): {a.grad:.4f}")
print(f"Gradient of g with respect to b (dg/db): {b.grad:.4f}")
```

### Acknowledgements

This project was inspired by and built following the insightful micrograd lecture from Andrej Karpathy's "Neural Networks: Zero to Hero" course. His clear explanations were instrumental in understanding these complex topics.

