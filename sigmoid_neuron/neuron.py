import math


class Neuron:
    """
    creates a Neuron object.
    :param weights: takes in a list of floats as weights for the neuron.
    :param bias: takes in a float as bias for the neuron.
    :param name: takes in a string for the name of the neuron.
    """
    def __init__(self, weights, bias, name):
        self.weights = weights
        self.bias = bias
        self.name = name

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}"

    def sigmoid_activation_function(self, weighted_sum):
        """
        :param weighted_sum: gets a float and inputs it into the sigmoid function.
        :return: returns a value between 0 and 1.
        """
        return 1/(1+math.e**-weighted_sum)

    def activate(self, inputs):
        """
        :param inputs: gets a list of ints and calculates the weighted sum based on the weights and the bias.
        :return: returns a sigmoid_activation_function call with weighted_sum as argument.
        """
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum = weighted_sum + self.bias
        return self.sigmoid_activation_function(weighted_sum)
