# __Assisted Practice: Configuring Neural Network and Activation Function__
- Configuring a neural network involves setting various parameters and architectural choices to define the structure and behavior of the network.
Let's understand how to configure neural network in Python.

- An activation function determines the output of a neuron based on the weighted sum of its inputs, introducing non-linearity and enabling complex modeling in neural networks.
Let's understand how to build a simple neural network in Python, considering the activation function as tanh.

Let's understand how to build a perceptron-based classification model.

## Steps to be followed:
1. Import the required libraries
2. Initialize the weights
3. Update the weights
4. Initialize the think function and neural network
5. Train the neural network

### Step 1: Import the required libraries

- Import the necessary modules for numerical computations and define functions for exponential calculations, array operations, random number generation, and matrix multiplication.

import numpy as np
from numpy import exp, array, random, dot

### Step 2: Initialize the weights

- Define a class with the name __NeuralNetwork__.
- Seed the random number generator so it generates the same numbers every time the program runs.
- Assign random weights to a 3X1 matrix, with values in the range __-1__ to __1__ and a mean of __0__.
- Use the __tanh__ function to describe an S-shaped curve.



### Step 3: Update the weights
- Train the neural network through a process of trial and error.
- Adjust the synaptic weights each time.
- Pass the training set through a neural network (a single neuron).
-  Calculate the error (the difference between the desired and predicted outputs).
- Adjust the weights.




class NeuralNetwork():
    def __init__(self):
        np.random.seed(2)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_derivative(self, x):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return 1 - t**2

    def __train__(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = np.dot(training_set_inputs.T, error * self.__tanh_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__tanh(np.dot(inputs, self.synaptic_weights))

### Step 4: Initialize the think function and neural network
- The __think__ function calculates the dot product between an array of inputs and the neural network's synaptic weights. It then applies the hyperbolic tangent activation function, __tanh__, to the result and returns the output.
- The main code block creates an instance of the __NeuralNetwork__ class and prints the initial random values of the synaptic weights.
- The **training_set_inputs** variable represents the input data for training the neural network.
- The training set input is a 2D array where each row corresponds to a set of input values.
- The **training_set_outputs** variable represents the corresponding output values for the training set.
- The training set output is a 2D array where each row corresponds to the expected output for the corresponding input set.

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights:")
    print(neural_network.synaptic_weights)

    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

**Observations:**
- The random starting synaptic weights are initialized with three values represented as a 2D array.
- Each value in the array corresponds to the synaptic weight connecting the input neurons to the single output neuron of the neural network.
- The values of the synaptic weights are randomly generated.
- In this case, the weights are approximately **-0.12801**, **-0.94814**, and **0.09932**.



### Step 5: Train the neural network
- Train the neural network using training sets.
- Perform training sets __10,000__ times and make small adjustments each time.
- Test the neural network with a new situation.

    neural_network.__train__(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training:")
    print(neural_network.synaptic_weights)

    print("Considering new situation [1, 0, 0] -> ?:")
    print(neural_network.think(np.array([1, 0, 0])))

**Observations:**

- After training, the synaptic weights of the neural network are updated.
- The updated synaptic weights are approximately __5.74063__, __-0.19473__, and __-0.34309__.
- The neural network is then provided with the new inputs of __1__, __0__, and __0__.
- The predicted output by the neural network for these inputs is approximately __0.99995__.
- The output represents the result of passing the inputs through the neural network after training, indicating the network's prediction or response.
