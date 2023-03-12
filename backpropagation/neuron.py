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
        self.error = None

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}, error = {self.error}"

    def sigmoid(self, sig_input):
        """
        :param sig_input: gets a float and inputs it into the sigmoid function.
        :return: returns a value between 0 and 1.
        """
        return 1/(1+math.e**-sig_input)

    def sigmoid_deriv(self, sig_input):
        """
        :param sig_input: gets a float and inputs it into the sigmoid derivative function.
        :return: returns a value between 0 and 1.
        """
        return self.sigmoid(sig_input)*(1-self.sigmoid(sig_input))

    def calculate_error(self, inputs, activation):
        output = self.activate(inputs)
        inasdf = []
        for i in range(len(inputs)):
            ine = self.sigmoid_deriv(inputs[i]) * -(activation - output)
            inasdf.append(ine)

        print(inasdf)
        # j =
        pass


    def update(self, inputs, activation):
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
        return self.sigmoid(weighted_sum)
