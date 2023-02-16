import numpy as np


class Perceptron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.bias = -threshold

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}"

    def activate(self, inputs):
        wighted_sum = 0
        for i in range(len(inputs)):
            wighted_sum += inputs[i] * self.weights[i]
            # print(inputs[i])
            # print(self.weights[i])
            # print(inputs[i] * self.weights[i])
            # print("")
        print(wighted_sum)
        print(wighted_sum + self.bias)
        if wighted_sum + self.bias > 0:
            return 1
        else:
            return 0


p1 = Perceptron([.7, 0.3, .2, .8], 1.5)
print(p1)

p1_activate = p1.activate([1, 0, 1, 1])
print(p1_activate)
