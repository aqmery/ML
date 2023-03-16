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
        self.eta = .5
        self.error = None
        self.output = None

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}, error = {self.error}"

    def sigmoid(self, sig_input):
        """
        :param sig_input: gets a float and inputs it into the sigmoid function.
        :return: returns a value between 0 and 1.
        """
        return 1/(1+math.e**-sig_input)

    def calculate_error(self, inputs, target):
        output = self.activate(inputs)
        error = (output*(1-output))*-(target-output)
        self.error = error
        self.output = output
        print("self.output", self.output)
        print("self.error", self.error)
        print("target", target)
        print("")

    def calculate_gradient(self, weight):
        return weight*self.error

    def calculate_delta(self):
        weight_change = []
        for i in range(len(self.weights)):
            weight_change.append(self.eta*(self.weights[i]*self.error))
        bias_change = self.eta*self.error
        return weight_change, bias_change

    def update(self):
        weight_change, bias_change = self.calculate_delta()
        # print("weight_change", weight_change)
        # print("bias_change", bias_change)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-weight_change[i]
        self.bias = self.bias-bias_change


    def activate(self, inputs):
        """
        :param inputs: gets a list of ints and calculates the weighted sum based on the weights and the bias.
        :return: returns a sigmoid_activation_function call with weighted_sum as argument.
        """
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum = weighted_sum + self.bias
        output = self.sigmoid(weighted_sum)
        self.output = output
        return output
