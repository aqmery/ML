import math
from scipy.special import expit


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
        self.eta = .1
        self.error = 0

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}, error = {self.error}"

    def sigmoid_activation_function(self, weighted_sum):
        """
        :param weighted_sum: gets a float and inputs it into the sigmoid function.
        :return: returns a value between 0 and 1.
        """
        return 1/(1+math.e**-weighted_sum)

    # def error(self, inputs):
    #     weight_error = []
    #     for i_input in range(len(inputs)):
    #         temp_error = round(expit(sum([self.weights[i] * inputs[i_input][i] for i in range(len(self.weights))]) + self.bias), 3)
    #         weight_error.append(temp_error)
    #     self.error =


    def update(self, n_input, activation):
        pass


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
