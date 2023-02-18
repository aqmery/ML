class Perceptron:
    def __init__(self, weights, bias, name):
        self.weights = weights
        self.bias = bias
        self.name = name

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}"

    def step_function(self, weighted_sum):
        if weighted_sum + self.bias >= 0:
            self.returns = 1
            return 1
        else:
            self.returns = 0
            return 0

    def activate(self, inputs):
        weighted_sum = 0
        if type(inputs[0]) == list:
            # print(inputs)
            temp = []
            for i in range(len(inputs)):
                temp.append(inputs[i][0])
            inputs = temp
            # print(inputs)
        # print(inputs)
        # print(self.weights)
        # print(self.name)
        # print("")
        for i in range(len(inputs)):
            # print("i =", i)
            # print(inputs[i])
            # print("type(inputs[i]) =", type(inputs[i]))
            # print("type(self.weights[i]) =", type(self.weights[i]))
            # print("input * weight =", inputs[i] * self.weights[i])
            # print("----")
            weighted_sum += inputs[i] * self.weights[i]
        # print("")
        # print("")
        return self.step_function(weighted_sum)