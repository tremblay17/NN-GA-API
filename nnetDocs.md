# Perceptron Methods Documentation
## Constructor: init
The provided Python code is part of a neural network implementation, specifically the initialization of the weights and thresholds for the hidden layers and the output layer of the network.

The code begins with a nested loop that iterates over each node in each hidden layer. For each node, it initializes a list of weights and a threshold. The weights and thresholds are randomly initialized using a uniform distribution between -2.4 divided by the number of inputs and 2.4 divided by the number of inputs. This range is a common choice for initial weights in neural networks, as it helps to avoid saturation of activation functions at the start of training. The weights for each node are stored in self.weights, and the thresholds for each layer are stored in self.thresholds.

Next, the code initializes the weights and thresholds for the output layer in a similar manner. The number of inputs to the output layer is the number of nodes in the last hidden layer, which is why hiddenLayers[-1] is used in the loop.

Finally, the code restructures the self.weights list. It groups the weights for each hidden layer into sublists, and appends the weights for the output layer to the end. This makes it easier to access the weights for a specific layer later on. The index variable is used to keep track of the start of the weights for each layer in the original flat list.
###
## Activation Functions
### Linear
### Step
### Sigmoid
### Tanh
### ReLU
### Leaky ReLU
###
## Weight Training & Correction
### Summation
### Error Calculation & Gradient
### Delta Rule
### Sum of Squared Errors
###
## Training
###
## Test
