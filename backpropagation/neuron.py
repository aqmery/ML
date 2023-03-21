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
        """
        calculates the error of the input.
        :param inputs: the input.
        :param target: the target.
        :return: sets the self.error and self.output values.
        """
        output = self.activate(inputs)
        error = (output*(1-output))*-(target-output)
        self.error = error
        self.output = output

    def calculate_gradient(self, weight_out):
        """
        calculates the gradient of the weight.
        :param weight_out: the weight that gets calculated.
        :return: outputs the result.
        """
        return weight_out*self.error

    def calculate_delta(self, eta):
        """
        calculates the change of the weights and bias.
        :param eta: the "stepsize" of the changes.
        :return: the amount of change for each weight and bias.
        """
        weight_change = []
        for i in range(len(self.weights)):
            weight_change.append(eta*(self.inputs[i]*self.error))
        bias_change = eta*self.error
        return weight_change, bias_change

    def update(self, inputs, target, eta):
        """
        activates the neuron, calculates the delta of the wieghts and bias,
        updates the wieghts and bias of the neuron.
        :param inputs: inputs needed to activate the neuron.
        :param target: the target of those inputs.
        :param eta: the "stepsize" of the changes.
        :return: sets the self.weights and self.bias variables of the neuron.
        """
        self.calculate_error(inputs, target)
        weight_change, bias_change = self.calculate_delta(eta)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-weight_change[i]
        self.bias = self.bias-bias_change

    def hidden_update(self):
        """
        the "hidden" version of the normal "update" function, this is used in backpropagation.
        :return: sets the self.weights and self.bias variables of the neuron.
        """
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-self.weightchanges[i]
        self.bias = self.bias-self.biaschange


    def hidden_error(self, weights, error, eta):
        """
        the "hidden" version of the normal "calculate_error" function, this is used in backpropagation.
        :param weights: the weights from which the error is to be calculated.
        :param error: the error of the neuron located above it.
        :param eta: the "stepsize" of the changes.
        :return: returns the self.error and self.weights variables.
        """
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
