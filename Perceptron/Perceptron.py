class Perceptron:
    """
    creates a Perceptron object.
    :param weights: takes in a list of floats as weights for the perceptron.
    :param bias: takes in a float as bias for the perceptron.
    :param name: takes in a string for the name of the perceptron.
    """
    def __init__(self, weights, bias, name):
        self.weights = weights
        self.bias = bias
        self.name = name

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}"

    def step_function(self, weighted_sum):
        """
        :param weighted_sum: gets a float and checks if it's higher than 0.
        :return: returns 1 or 0 based on the weighted_sum.
        """
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def activate(self, inputs):
        """
        :param inputs: gets a list of ints and calculates the weighted sum based on the weights and the bias.
        :return: returns a step_function call with weighted_sum as argument.
        """
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum = weighted_sum + self.bias
        return self.step_function(weighted_sum)
