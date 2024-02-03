import numpy as np

#Creating a Neural Network to crack the famous XOr Problem

class Layer:
    def __init__(self):
        # Initialize input and output to None
        self.input = None
        self.output = None

    def forward(self, input):
        # Forward prop function
        # input: input data
        # output: computed output
        pass

    def backward(self, output_gradient, learning_rate):
        # Backward prop function
        # output_gradient: gradient of the loss with respect to the layer's output
        # learning_rate: learning rate for parameter updates
        # returns: gradient of the loss with respect to the layer's input
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initialize weights and biases with random values
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # Perform forward prop
        # input: input data
        # returns: computed output
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Perform backward prop
        # output_gradient: gradient of the loss with respect to the layer's output
        # learning_rate: learning rate for parameter updates
        # returns: gradient of the loss with respect to the layer's input

        # Compute the gradient of weights
        weights_gradient = np.dot(output_gradient, self.input.T)

        # Compute the gradient of bias
        # Note: We use np.sum() to sum the gradients across the batch, axis=1 sums across rows
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        # Compute the gradient of the loss with respect to the input of the layer
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Update weights and bias using gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        # Initialize activation functions and their derivatives
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        # Perform forward prop using activation function
        # input: input data
        # returns: computed output
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # Perform backward prop using derivative of activation function
        # output_gradient: gradient of the loss with respect to the layer's output
        # learning_rate: learning rate for parameter updates
        # returns: gradient of the loss with respect to the layer's input
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        # Initialize tanh activation function and its derivative
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

# Define mean squared error loss function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# Define the derivative of mean squared error loss function
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

# Make predictions using the given network
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# Train the neural network using backpropagation
def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # Forward prop
            output = predict(network, x)

            # Compute error
            error += loss(y, output)

            # Backward prop
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

# Define input and output data
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Define the neural network architecture
network = [
    Dense(2, 3),  # Input layer with 2 inputs and 3 neurons
    Tanh(),       # Tanh activation function
    Dense(3, 1),  # Hidden layer with 3 neurons connected to 1 neuron
    Tanh()        # Tanh activation function
]

# train
train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

#predict XOr for 0,0: should print 0
print(round(float(predict(network,[[0],[0]]))))

#predict XOr for 1,0: should print 1
print(round(float(predict(network,[[1],[0]]))))

#predict XOr for 0,1: should print 1
print(round(float(predict(network,[[0],[1]]))))

#predict XOr for 1,1: should print 0
print(round(float(predict(network,[[1],[1]]))))