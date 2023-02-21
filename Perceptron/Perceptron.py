class Perceptron:
    def __init__(self, weights, bias, name):
        self.weights = weights
        self.bias = bias
        self.name = name

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}"

    def step_function(self, weighted_sum):
        if weighted_sum >= 0:
            self.returns = 1
            return 1
        else:
            self.returns = 0
            return 0

    def activate(self, inputs):
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum = weighted_sum + self.bias
        return self.step_function(weighted_sum)