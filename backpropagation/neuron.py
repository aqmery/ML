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
        self.inputs = None
        self.error = None
        self.output = None
        self.weightchanges = None
        self.biaschange = None

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

    def calculate_gradient(self, weight_out):
        return weight_out*self.error

    def calculate_delta(self, eta):
        weight_change = []
        for i in range(len(self.weights)):
            weight_change.append(eta*(self.inputs[i]*self.error))
        bias_change = eta*self.error
        return weight_change, bias_change

    def update(self, inputs, target, eta):
        self.calculate_error(inputs, target)
        weight_change, bias_change = self.calculate_delta(eta)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-weight_change[i]
        self.bias = self.bias-bias_change

    def hidden_error(self, weights, error, eta):
        h_error = (self.output * (1 - self.output))
        sum_previous_layer = 0
        for i in range(len(error)):
            sum_previous_layer += weights[i]*error[i]
        self.error = h_error*sum_previous_layer
        weight_change = []
        for i in range(len(self.weights)):
            weight_change.append(eta * self.inputs[i]*self.error)
        bias_change = eta*1*self.error
        self.weightchanges = weight_change
        self.biaschange = bias_change
        print(weight_change)
        print(bias_change)
        print("")
        return self.error, self.weights

    def activate(self, inputs):
        """
        :param inputs: gets a list of ints and calculates the weighted sum based on the weights and the bias.
        :return: returns a sigmoid_activation_function call with weighted_sum as argument.
        """
        self.inputs = inputs
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum = weighted_sum + self.bias

        output = self.sigmoid(weighted_sum)
        self.output = output
        return output
