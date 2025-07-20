# From Scalar Autograd to a Custom Tensor Framework

My journey into the complex world of deep learning got really exciting when I decided to see how it all truly worked and build a deep learning system myself from the very beginning. This project wasn't just about writing code for rules; it was about really exploring the basic math and computer ideas that make today's AI work.

**Phase 1: Autograd with Scalar Operations (Inspired by Micrograd)**

My first idea came from a really helpful video by Andrej Karpathy. He explained how a computer can automatically figure out how much each small change affects the final answer. This made me want to build that core idea myself. So, starting from the very basics, I made a simple version of this autograd system.

This first part was super important. I carefully made a "Value" class, which was like the smallest building block of my math map. Each "Value" number held its actual number, but also its gradient, a link to the "Value" numbers that came before it, and what math step made it. This let me:

- Construct a Dynamic Computational Graph: Every arithmetic operation (+, *, exp, ln, pow, etc.) on Value objects automatically added nodes and edges to this graph, tracking dependencies.

- Implement the Chain Rule: The heart of backpropagation. For each operation, I defined a _backward() method. When called, this method applied the chain rule, multiplying the incoming gradient (out.grad) by the local derivative of the operation with respect to its inputs, and accumulating these contributions onto the grad of the parent Values. This solidified my understanding of how gradients flow backward through complex computations.

- Master Topological Sort: A critical component for ensuring correct gradient propagation. Before the backward pass, I implemented a topological sort algorithm to order the Value objects. This ensured that the _backward() method of a Value was only called after all gradients from its dependent (child) operations had been computed and accumulated, guaranteeing accurate gradient updates.

With this scalar autograd in place, I successfully built and trained a small neural network. It was incredibly rewarding to see gradients flow and the model learn, but a significant challenge quickly emerged: speed.

**Phase 2: Autograd with Tensors and Matrix Calculus**

While that simple autograd system was great for learning, it was just too slow for real AI jobs, which use huge matrices. My next big goal was to make learning much, much faster by building a new autograd system based on Tensors.

This was a big jump into harder stuff. It meant I really needed to understand "matrix calculus" – which is just how you figure out the gradients for operations when you're working with matrices, not just scalars. I dug deep into the tricky parts of:

- Derivatives of matrix multiplication.

- Gradients of summation and reduction operations across dimensions.

- Broadcasting rules and their impact on gradient shapes.

My new Tensor class was designed to handle multi-dimensional arrays, with operations like exp(), sum(), max(), and element-wise arithmetic now working across entire tensors. I also integrated features common in production frameworks:

- requires_grad: Allowing selective tracking of gradients, crucial for memory optimization and distinguishing trainable parameters from constant inputs.

- Optimized Operations: Implementing sum(), max(), and other reduction/element-wise methods to correctly compute values and set up their corresponding _backward() functions for tensor gradients.

**Phase 3: Problem with Broadcasting**

The transition to tensors wasn't without challenges. I hit a significant roadblock when dealing with NumPy broadcasting (or the equivalent logic in a custom tensor implementation). Operations like adding a scalar to a matrix, or adding matrices of different but compatible shapes, require careful handling of gradient propagation. I had to thoroughly understand:

What broadcasting is: How NumPy implicitly expands dimensions to make shapes compatible for element-wise operations.

How it affects gradients: When backpropagating through a broadcasted operation, gradients need to be correctly summed or "reduced" along the dimensions that were broadcasted in the forward pass. This required careful implementation of _backward() methods for operations involving broadcasting.

Learning and fixing this specific issue was a testament to the iterative nature of software development and problem-solving in deep learning.

**Phase 4: Training Both Models**

Finally, I put both my scalar and tensor frameworks to the test by training classification models custom datasets. Here are the results:

- Scalar Model: Trained for 20 epochs, it took approximately 10 minutes and 51.86 seconds.

- Tensor Model: Trained for 100 epochs (5x more iterations), it completed in just 2.36 seconds.

This showed a huge jump in how fast the AI could learn. It proved how much power and speed you get by moving from working with scalars to working with tensors. Plus, the tensor-based AI actually learned just as well, or even better, in a tiny fraction of the time. This showed that being fast doesn't mean you have to be less accurate.

This project was a super valuable experience. It really made me understand how AI learns automatically, how math steps are mapped out, and the small tricks that make AI programs run fast. It gave me a deep respect for how clever and complicated today's AI systems are. It also made me much better at solving problems on my own, using math, and making things run faster. I'm really excited to use these basic ideas to take on even harder AI challenges.
